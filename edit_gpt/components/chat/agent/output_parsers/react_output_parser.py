import ast
import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)
MISSING_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Before the Action, there must be a Thought!"
)
INVALID_INPUT_ERROR_MESSAGE = "Invalid Action Input. Could not parse tool_input."
NONE_IN_INPUT_ERROR_MESSAGE = (
    "Could not parse tool_input. None key or value found in tool_input."
)


class ReActSingleInputOutputDictParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if re.match(r"^\s*Action", text):
            raise OutputParserException(
                f"Invalid LLM output: {text}. Before the Action, there must be a Thought!",
                observation=MISSING_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')
            try:
                tool_input_dict = ast.literal_eval(tool_input)
            except SyntaxError:
                raise OutputParserException(
                    f"Invalid LLM output: {text}. Could not parse tool_input.",
                    observation=INVALID_INPUT_ERROR_MESSAGE,
                    llm_output=text,
                    send_to_llm=True,
                )
            if not tool_input_dict:
                raise OutputParserException(
                    f"Invalid LLM output: {text}. None key or value found in tool_input.",
                    observation=NONE_IN_INPUT_ERROR_MESSAGE,
                    llm_output=text,
                    send_to_llm=True,
                )

            for key, value in tool_input_dict.items():
                if key is None or value is None:
                    raise OutputParserException(
                        f"Invalid LLM output: {text}. None key or value found in tool_input.",
                        observation=NONE_IN_INPUT_ERROR_MESSAGE,
                        llm_output=text,
                        send_to_llm=True,
                    )

            if (
                not includes_answer
                or includes_answer
                and text.index(FINAL_ANSWER_ACTION) > text.index("Action")
            ):
                return AgentAction(action, tool_input_dict, text)
            else:
                return AgentFinish(
                    {
                        "output": text.split(FINAL_ANSWER_ACTION)[-1]
                        .split("\n", 1)[0]
                        .strip()
                    },
                    text,
                )

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: {text}",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: {text}",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: {text}")

    @property
    def _type(self) -> str:
        return "react-single-input"
