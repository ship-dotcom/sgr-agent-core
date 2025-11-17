"""Prompt-based SGR Agent that uses prompt instructions for schema definition.

This agent implements the same reasoning flow as SGRAgent but uses prompt-based
instructions to ensure output format instead of the response_format parameter.
This makes it compatible with models that don't support structured outputs.
"""

import json
import logging
import re
from typing import Type

from openai import AsyncOpenAI

from sgr_deep_research.core.agent_definition import ExecutionConfig, LLMConfig, PromptsConfig
from sgr_deep_research.core.base_agent import BaseAgent
from sgr_deep_research.core.tools import (
    BaseTool,
    ClarificationTool,
    CreateReportTool,
    FinalAnswerTool,
    NextStepToolsBuilder,
    NextStepToolStub,
    WebSearchTool,
)

logger = logging.getLogger(__name__)


class PromptBasedSGRAgent(BaseAgent):
    """Agent for deep research tasks using SGR framework with prompt-based schema definition.

    This agent uses prompts to instruct the LLM on the expected output format,
    making it compatible with models that don't support the response_format parameter.
    """

    name: str = "prompt_based_sgr_agent"

    def __init__(
        self,
        task: str,
        openai_client: AsyncOpenAI,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        execution_config: ExecutionConfig,
        toolkit: list[Type[BaseTool]] | None = None,
    ):
        super().__init__(
            task=task,
            openai_client=openai_client,
            llm_config=llm_config,
            prompts_config=prompts_config,
            execution_config=execution_config,
            toolkit=toolkit,
        )
        self.max_searches = execution_config.max_searches

    def _get_tools_schema(self, tools: list[Type[BaseTool]]) -> str:
        """Generate a schema description for available tools in a human-readable format."""
        schema_parts = []
        for tool in tools:
            # Get the JSON schema for the tool
            tool_schema = tool.model_json_schema()
            properties = tool_schema.get("properties", {})
            required = tool_schema.get("required", [])

            # Format tool description
            tool_desc = f"Tool: {tool.tool_name}\n"
            tool_desc += f"Description: {tool.description}\n"
            tool_desc += "Parameters:\n"

            for prop_name, prop_info in properties.items():
                if prop_name == "tool_name_discriminator":
                    continue
                prop_type = prop_info.get("type", "string")
                prop_desc = prop_info.get("description", "")
                is_required = " (required)" if prop_name in required else " (optional)"
                tool_desc += f"  - {prop_name} ({prop_type}){is_required}: {prop_desc}\n"

            schema_parts.append(tool_desc)

        return "\n".join(schema_parts)

    async def _prepare_context(self) -> list[dict]:
        """Prepare conversation context with system prompt including tool schemas."""
        from sgr_deep_research.core.services.prompt_loader import PromptLoader
        import os
        from pathlib import Path

        # Get the tools that would be available
        tools = await self._get_available_tools()

        # Generate schema for tools
        tools_schema = self._get_tools_schema(tools)

        # Get available tools description
        available_tools_str_list = [
            f"{i}. {tool.tool_name}: {tool.description}" for i, tool in enumerate(tools, start=1)
        ]

        # Load the prompt-based system prompt template
        try:
            # Find the prompts directory relative to this file
            prompts_dir = Path(__file__).parent.parent / "prompts"
            template_path = prompts_dir / "prompt_based_system_prompt.txt"

            with open(template_path, "r") as f:
                template = f.read()

            system_prompt = template.format(
                tools_schema=tools_schema,
                available_tools="\n".join(available_tools_str_list),
            )
        except Exception as e:
            logger.warning(f"Failed to load prompt-based system prompt: {e}, falling back to default")
            system_prompt = PromptLoader.get_system_prompt(self.toolkit, self.prompts_config)

        return [
            {"role": "system", "content": system_prompt},
            *self.conversation,
        ]

    async def _get_available_tools(self) -> list[Type[BaseTool]]:
        """Get list of available tools based on current context."""
        tools = set(self.toolkit)
        if self._context.iteration >= self.max_iterations:
            tools = {
                CreateReportTool,
                FinalAnswerTool,
            }
        if self._context.clarifications_used >= self.max_clarifications:
            tools -= {
                ClarificationTool,
            }
        if self._context.searches_used >= self.max_searches:
            tools -= {
                WebSearchTool,
            }
        return list(tools)

    async def _prepare_tools(self) -> Type[NextStepToolStub]:
        """Prepare tool classes with current context limits."""
        tools = await self._get_available_tools()
        return NextStepToolsBuilder.build_NextStepTools(tools)

    def _parse_tool_call_from_response(self, response_text: str) -> dict:
        """Parse tool call from LLM response text.

        Expected format:
        <tool_call>
        {"name": "tool_name", "arguments": {...}}
        </tool_call>
        """
        # Extract content between <tool_call> tags
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if not matches:
            raise ValueError(f"No tool call found in response: {response_text[:200]}")

        # Parse the JSON
        try:
            tool_call_json = json.loads(matches[0])
            return tool_call_json
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse tool call JSON: {e}, content: {matches[0][:200]}")

    async def _reasoning_phase(self) -> NextStepToolStub:
        """Execute reasoning phase using prompt-based tool calling."""
        async with self.openai_client.chat.completions.stream(
            model=self.llm_config.model,
            messages=await self._prepare_context(),
            max_tokens=self.llm_config.max_tokens,
            temperature=self.llm_config.temperature,
        ) as stream:
            async for event in stream:
                if event.type == "chunk":
                    self.streaming_generator.add_chunk(event.chunk)

        # Get the complete response
        completion = await stream.get_final_completion()
        response_text = completion.choices[0].message.content

        if not response_text:
            raise ValueError("Empty response from LLM")

        # Parse tool call from response
        tool_call_data = self._parse_tool_call_from_response(response_text)

        # Build the NextStepTool type dynamically
        NextStepTool = await self._prepare_tools()

        # Extract tool name and arguments
        tool_name = tool_call_data.get("name")
        tool_arguments = tool_call_data.get("arguments", {})

        # Create the reasoning object with the tool function
        # We need to reconstruct the full object that matches NextStepTool schema
        reasoning_data = {
            **tool_arguments,
            "function": {
                "tool_name_discriminator": tool_name,
                **tool_arguments.get("function", {}),
            },
        }

        try:
            reasoning: NextStepToolStub = NextStepTool.model_validate(reasoning_data)
        except Exception as e:
            logger.error(f"Failed to validate reasoning: {e}, data: {reasoning_data}")
            raise

        self._log_reasoning(reasoning)
        return reasoning

    async def _select_action_phase(self, reasoning: NextStepToolStub) -> BaseTool:
        """Select action tool from reasoning result."""
        tool = reasoning.function
        if not isinstance(tool, BaseTool):
            raise ValueError("Selected tool is not a valid BaseTool instance")

        self.conversation.append(
            {
                "role": "assistant",
                "content": reasoning.remaining_steps[0] if reasoning.remaining_steps else "Completing",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": f"{self._context.iteration}-action",
                        "function": {
                            "name": tool.tool_name,
                            "arguments": tool.model_dump_json(),
                        },
                    }
                ],
            }
        )
        self.streaming_generator.add_tool_call(
            f"{self._context.iteration}-action", tool.tool_name, tool.model_dump_json()
        )
        return tool

    async def _action_phase(self, tool: BaseTool) -> str:
        """Execute the selected tool."""
        result = await tool(self._context)
        self.conversation.append(
            {"role": "tool", "content": result, "tool_call_id": f"{self._context.iteration}-action"}
        )
        self.streaming_generator.add_chunk_from_str(f"{result}\n")
        self._log_tool_execution(tool, result)
        return result
