import json
import logging
import re
from typing import Literal, Type

from openai import AsyncOpenAI, pydantic_function_tool
from openai.types.chat import ChatCompletionFunctionToolParam
from pydantic import ValidationError

from sgr_agent_core.agent_config import AgentConfig
from sgr_agent_core.agents.sgr_agent import SGRAgent
from sgr_agent_core.models import AgentStatesEnum
from sgr_agent_core.tools import (
    BaseTool,
    ClarificationTool,
    CreateReportTool,
    ExtractPageContentTool,
    FinalAnswerTool,
    ReasoningTool,
    WebSearchTool,
)

logger = logging.getLogger(__name__)


class SGRToolCallingAgent(SGRAgent):
    """Agent that uses OpenAI native function calling to select and execute
    tools based on SGR like a reasoning scheme."""

    name: str = "sgr_tool_calling_agent"

    def __init__(
        self,
        task: str,
        openai_client: AsyncOpenAI,
        agent_config: AgentConfig,
        toolkit: list[Type[BaseTool]],
        def_name: str | None = None,
        **kwargs: dict,
    ):
        super().__init__(
            task=task,
            openai_client=openai_client,
            agent_config=agent_config,
            toolkit=toolkit,
            def_name=def_name,
            **kwargs,
        )
        self.tool_choice: Literal["required"] = "required"

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean JSON string by removing trailing characters after the last closing brace.
        
        Args:
            json_str: Raw JSON string that may contain trailing characters
            
        Returns:
            Cleaned JSON string
        """
        # Remove leading/trailing whitespace
        json_str = json_str.strip()
        
        # Find the last closing brace or bracket
        last_brace = json_str.rfind('}')
        last_bracket = json_str.rfind(']')
        last_pos = max(last_brace, last_bracket)
        
        if last_pos > 0:
            # Extract JSON up to the last closing brace/bracket
            cleaned = json_str[:last_pos + 1]
            
            # Try to parse to validate
            try:
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError:
                pass
        
        # If that didn't work, try to find JSON object boundaries
        # Look for first { and last }
        first_brace = json_str.find('{')
        if first_brace >= 0 and last_brace > first_brace:
            cleaned = json_str[first_brace:last_brace + 1]
            try:
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError:
                pass
        
        # Last resort: try to extract JSON using regex
        # Match JSON object: { ... }
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, json_str)
        if match:
            return match.group(0)
        
        # If all else fails, return original (will fail with better error message)
        return json_str

    async def _reasoning_phase(self) -> ReasoningTool:
        async with self.openai_client.chat.completions.stream(
            messages=await self._prepare_context(),
            tools=[pydantic_function_tool(ReasoningTool, name=ReasoningTool.tool_name)],
            tool_choice={"type": "function", "function": {"name": ReasoningTool.tool_name}},
            **self.config.llm.to_openai_client_kwargs(),
        ) as stream:
            async for event in stream:
                if event.type == "chunk":
                    self.streaming_generator.add_chunk(event.chunk)
            
            completion = await stream.get_final_completion()
            try:
                reasoning: ReasoningTool = (
                    completion.choices[0].message.tool_calls[0].function.parsed_arguments
                )
            except (ValidationError, ValueError) as e:
                # JSON parsing error - try to fix trailing characters
                logger.warning(f"JSON parsing error in reasoning phase, attempting to fix: {e}")
                try:
                    tool_call = completion.choices[0].message.tool_calls[0]
                    raw_arguments = tool_call.function.arguments
                    cleaned_json = self._clean_json_string(raw_arguments)
                    json_data = json.loads(cleaned_json)
                    reasoning = ReasoningTool(**json_data)
                    logger.info("Successfully fixed JSON parsing for reasoning")
                except Exception as fix_error:
                    logger.error(f"Failed to fix JSON parsing in reasoning: {fix_error}", exc_info=True)
                    raise ValueError(f"Failed to parse reasoning arguments: {e}")
        self.conversation.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": f"{self._context.iteration}-reasoning",
                        "function": {
                            "name": reasoning.tool_name,
                            "arguments": reasoning.model_dump_json(),
                        },
                    }
                ],
            }
        )
        tool_call_result = await reasoning(self._context)
        self.streaming_generator.add_tool_call(
            f"{self._context.iteration}-reasoning", reasoning.tool_name, tool_call_result
        )
        self.conversation.append(
            {"role": "tool", "content": tool_call_result, "tool_call_id": f"{self._context.iteration}-reasoning"}
        )
        self._log_reasoning(reasoning)
        return reasoning

    async def _select_action_phase(self, reasoning: ReasoningTool) -> BaseTool:
        async with self.openai_client.chat.completions.stream(
            messages=await self._prepare_context(),
            tools=await self._prepare_tools(),
            tool_choice=self.tool_choice,
            **self.config.llm.to_openai_client_kwargs(),
        ) as stream:
            async for event in stream:
                if event.type == "chunk":
                    self.streaming_generator.add_chunk(event.chunk)

        completion = await stream.get_final_completion()

        try:
            tool = completion.choices[0].message.tool_calls[0].function.parsed_arguments
        except (IndexError, AttributeError, TypeError):
            # LLM returned a text response instead of a tool call - treat as completion
            final_content = completion.choices[0].message.content or "Task completed successfully"
            tool = FinalAnswerTool(
                reasoning="Agent decided to complete the task",
                completed_steps=[],
                answer=final_content,
                status=AgentStatesEnum.COMPLETED,
            )
        except (ValidationError, ValueError) as e:
            # JSON parsing error - try to fix trailing characters
            logger.warning(f"JSON parsing error, attempting to fix: {e}")
            try:
                tool_call = completion.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments
                
                # Find the tool class
                tool_class = None
                for tool_type in self.toolkit:
                    if tool_type.tool_name == function_name:
                        tool_class = tool_type
                        break
                
                if not tool_class:
                    raise ValueError(f"Tool class not found for {function_name}")
                
                # Try to extract valid JSON from raw_arguments
                # Remove trailing characters after the last closing brace
                cleaned_json = self._clean_json_string(raw_arguments)
                
                # Parse JSON and create tool instance
                json_data = json.loads(cleaned_json)
                tool = tool_class(**json_data)
                logger.info(f"Successfully fixed JSON parsing for {function_name}")
            except Exception as fix_error:
                logger.error(f"Failed to fix JSON parsing: {fix_error}", exc_info=True)
                raise ValueError(f"Failed to parse tool arguments for {function_name}: {e}")
        
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


class ResearchSGRToolCallingAgent(SGRToolCallingAgent):
    """Agent for deep research tasks."""

    def __init__(
        self,
        task: str,
        openai_client: AsyncOpenAI,
        agent_config: AgentConfig,
        toolkit: list[Type[BaseTool]],
        def_name: str | None = None,
        **kwargs: dict,
    ):
        research_toolkit = [WebSearchTool, ExtractPageContentTool, CreateReportTool, FinalAnswerTool]
        super().__init__(
            task=task,
            openai_client=openai_client,
            agent_config=agent_config,
            toolkit=research_toolkit + [t for t in toolkit if t not in research_toolkit],
            def_name=def_name,
            **kwargs,
        )

    async def _prepare_tools(self) -> list[ChatCompletionFunctionToolParam]:
        """Prepare available tools for the current agent state and progress."""
        tools = set(self.toolkit)
        if self._context.iteration >= self.config.execution.max_iterations:
            tools = {
                ReasoningTool,
                CreateReportTool,
                FinalAnswerTool,
            }
        if self._context.clarifications_used >= self.config.execution.max_clarifications:
            tools -= {
                ClarificationTool,
            }
        if self._context.searches_used >= self.config.search.max_searches:
            tools -= {
                WebSearchTool,
            }
        return [pydantic_function_tool(tool, name=tool.tool_name, description="") for tool in tools]
