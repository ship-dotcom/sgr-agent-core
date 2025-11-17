"""Agents module for SGR Agent Core."""

from sgr_deep_research.core.agents.prompt_based_sgr_agent import PromptBasedSGRAgent
from sgr_deep_research.core.agents.sgr_agent import SGRAgent
from sgr_deep_research.core.agents.sgr_auto_tool_calling_agent import SGRAutoToolCallingAgent
from sgr_deep_research.core.agents.sgr_so_tool_calling_agent import SGRSOToolCallingAgent
from sgr_deep_research.core.agents.sgr_tool_calling_agent import SGRToolCallingAgent
from sgr_deep_research.core.agents.tool_calling_agent import ToolCallingAgent

__all__ = [
    "PromptBasedSGRAgent",
    "SGRAgent",
    "SGRAutoToolCallingAgent",
    "SGRSOToolCallingAgent",
    "SGRToolCallingAgent",
    "ToolCallingAgent",
]
