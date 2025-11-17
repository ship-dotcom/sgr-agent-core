"""Tests for PromptBasedSGRAgent.

This module contains tests for the PromptBasedSGRAgent class that uses
prompt-based schema definition instead of response_format.
"""

from unittest.mock import Mock

import pytest

from sgr_deep_research.core.agent_definition import ExecutionConfig
from sgr_deep_research.core.agents.prompt_based_sgr_agent import PromptBasedSGRAgent
from sgr_deep_research.core.tools import (
    FinalAnswerTool,
    ReasoningTool,
    WebSearchTool,
)
from tests.conftest import create_test_agent


class TestPromptBasedSGRAgentInitialization:
    """Tests for PromptBasedSGRAgent initialization."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        agent = create_test_agent(
            PromptBasedSGRAgent,
            task="Test task",
            execution_config=ExecutionConfig(max_iterations=20, max_clarifications=3, max_searches=10),
        )

        assert agent.task == "Test task"
        assert agent.name == "prompt_based_sgr_agent"
        assert agent.max_iterations == 20
        assert agent.max_clarifications == 3
        assert agent.max_searches == 10

    def test_initialization_with_toolkit(self):
        """Test initialization with custom toolkit."""
        toolkit = [WebSearchTool, FinalAnswerTool]
        agent = create_test_agent(
            PromptBasedSGRAgent,
            task="Test",
            toolkit=toolkit,
        )

        assert WebSearchTool in agent.toolkit
        assert FinalAnswerTool in agent.toolkit


class TestPromptBasedSGRAgentToolSchema:
    """Tests for tool schema generation."""

    def test_get_tools_schema(self):
        """Test generating tool schema from tool classes."""
        agent = create_test_agent(PromptBasedSGRAgent, task="Test")

        tools = [ReasoningTool, WebSearchTool]
        schema = agent._get_tools_schema(tools)

        assert "Tool: reasoningtool" in schema
        assert "Tool: websearchtool" in schema
        assert "Description:" in schema
        assert "Parameters:" in schema

    def test_get_tools_schema_excludes_discriminator(self):
        """Test that tool schema excludes tool_name_discriminator."""
        agent = create_test_agent(PromptBasedSGRAgent, task="Test")

        tools = [ReasoningTool]
        schema = agent._get_tools_schema(tools)

        assert "tool_name_discriminator" not in schema


class TestPromptBasedSGRAgentToolSelection:
    """Tests for tool selection and preparation."""

    @pytest.mark.asyncio
    async def test_get_available_tools_basic(self):
        """Test getting available tools in basic state."""
        agent = create_test_agent(
            PromptBasedSGRAgent,
            task="Test",
            toolkit=[WebSearchTool, FinalAnswerTool, ReasoningTool],
        )

        tools = await agent._get_available_tools()

        assert WebSearchTool in tools
        assert FinalAnswerTool in tools
        assert ReasoningTool in tools

    @pytest.mark.asyncio
    async def test_get_available_tools_max_iterations_reached(self):
        """Test that only completion tools are available at max iterations."""
        agent = create_test_agent(
            PromptBasedSGRAgent,
            task="Test",
            toolkit=[WebSearchTool, FinalAnswerTool],
            execution_config=ExecutionConfig(max_iterations=5),
        )
        agent._context.iteration = 5

        tools = await agent._get_available_tools()

        assert FinalAnswerTool in tools
        assert WebSearchTool not in tools

    @pytest.mark.asyncio
    async def test_get_available_tools_max_searches_reached(self):
        """Test that search tools are removed when max searches reached."""
        from sgr_deep_research.core.tools import ClarificationTool

        agent = create_test_agent(
            PromptBasedSGRAgent,
            task="Test",
            toolkit=[WebSearchTool, ClarificationTool, FinalAnswerTool],
            execution_config=ExecutionConfig(max_searches=3),
        )
        agent._context.searches_used = 3

        tools = await agent._get_available_tools()

        assert WebSearchTool not in tools
        assert ClarificationTool in tools
        assert FinalAnswerTool in tools


class TestPromptBasedSGRAgentToolCallParsing:
    """Tests for parsing tool calls from LLM responses."""

    def test_parse_tool_call_basic(self):
        """Test parsing a basic tool call."""
        agent = create_test_agent(PromptBasedSGRAgent, task="Test")

        response = """
        Let me search for that information.
        <tool_call>
        {"name": "websearchtool", "arguments": {"query": "test query", "reasoning": "need info"}}
        </tool_call>
        """

        result = agent._parse_tool_call_from_response(response)

        assert result["name"] == "websearchtool"
        assert result["arguments"]["query"] == "test query"

    def test_parse_tool_call_with_whitespace(self):
        """Test parsing tool call with extra whitespace."""
        agent = create_test_agent(PromptBasedSGRAgent, task="Test")

        response = """
        <tool_call>
        {
            "name": "websearchtool",
            "arguments": {
                "query": "test query"
            }
        }
        </tool_call>
        """

        result = agent._parse_tool_call_from_response(response)

        assert result["name"] == "websearchtool"

    def test_parse_tool_call_no_tags(self):
        """Test that parsing fails when no tool_call tags are found."""
        agent = create_test_agent(PromptBasedSGRAgent, task="Test")

        response = '{"name": "websearchtool", "arguments": {"query": "test"}}'

        with pytest.raises(ValueError, match="No tool call found"):
            agent._parse_tool_call_from_response(response)

    def test_parse_tool_call_invalid_json(self):
        """Test that parsing fails with invalid JSON."""
        agent = create_test_agent(PromptBasedSGRAgent, task="Test")

        response = """
        <tool_call>
        {name: "websearchtool", arguments: {}}
        </tool_call>
        """

        with pytest.raises(ValueError, match="Failed to parse tool call JSON"):
            agent._parse_tool_call_from_response(response)

    def test_parse_tool_call_multiple_tags(self):
        """Test parsing when multiple tool_call tags are present (uses first)."""
        agent = create_test_agent(PromptBasedSGRAgent, task="Test")

        response = """
        <tool_call>
        {"name": "first_tool", "arguments": {}}
        </tool_call>
        Some text
        <tool_call>
        {"name": "second_tool", "arguments": {}}
        </tool_call>
        """

        result = agent._parse_tool_call_from_response(response)

        # Should parse the first tool call
        assert result["name"] == "first_tool"


class TestPromptBasedSGRAgentContextPreparation:
    """Tests for context preparation with prompt-based schema."""

    @pytest.mark.asyncio
    async def test_prepare_context_includes_tool_schema(self):
        """Test that prepared context includes tool schemas."""
        agent = create_test_agent(
            PromptBasedSGRAgent,
            task="Test",
            toolkit=[WebSearchTool, FinalAnswerTool],
        )

        context = await agent._prepare_context()

        # Should have system message
        assert len(context) >= 1
        assert context[0]["role"] == "system"

        # System message should contain tool information
        system_content = context[0]["content"]
        assert "tool_call" in system_content.lower() or "tools" in system_content.lower()

    @pytest.mark.asyncio
    async def test_prepare_context_with_conversation(self):
        """Test context preparation with existing conversation."""
        agent = create_test_agent(
            PromptBasedSGRAgent,
            task="Test",
            toolkit=[WebSearchTool],
        )
        agent.conversation = [
            {"role": "user", "content": "test message"},
            {"role": "assistant", "content": "test response"},
        ]

        context = await agent._prepare_context()

        # Should include system + conversation
        assert len(context) == 3
        assert context[0]["role"] == "system"
        assert context[1]["role"] == "user"
        assert context[2]["role"] == "assistant"


class TestPromptBasedSGRAgentIntegration:
    """Integration tests for PromptBasedSGRAgent phases."""

    @pytest.mark.asyncio
    async def test_select_action_phase(self):
        """Test select action phase with a valid reasoning result."""
        agent = create_test_agent(
            PromptBasedSGRAgent,
            task="Test",
            toolkit=[FinalAnswerTool],
        )

        # Create a mock reasoning result
        reasoning = Mock(spec=ReasoningTool)
        reasoning.function = FinalAnswerTool(
            reasoning="Test complete",
            completed_steps=["Step 1"],
            answer="Final answer",
            status="completed",
        )
        reasoning.remaining_steps = ["Complete task"]

        tool = await agent._select_action_phase(reasoning)

        assert isinstance(tool, FinalAnswerTool)
        assert len(agent.conversation) == 1
        assert agent.conversation[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_action_phase(self):
        """Test action phase execution."""
        agent = create_test_agent(PromptBasedSGRAgent, task="Test")
        agent._context.iteration = 1

        # Create a real tool instance for testing
        from sgr_deep_research.core.models import AgentStatesEnum

        tool = FinalAnswerTool(
            reasoning="Test execution",
            completed_steps=["Step 1"],
            answer="Test answer",
            status=AgentStatesEnum.COMPLETED,
        )

        result = await agent._action_phase(tool)

        # Should be JSON string
        assert "Test answer" in result
        assert len(agent.conversation) == 1
        assert agent.conversation[0]["role"] == "tool"
