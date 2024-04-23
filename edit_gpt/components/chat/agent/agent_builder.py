from typing import List

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.tools import BaseTool

from edit_gpt.components.chat.agent.agent import RunnableAgentWithStop
from edit_gpt.components.chat.agent.agent_prompts import BASE_REACT_PROMPT
from edit_gpt.components.chat.agent.output_parsers.react_output_parser import (
    ReActSingleInputOutputDictParser,
)


def initialize_agent(
    chat_model: BaseChatModel,
    tools: List[BaseTool],
    agent_max_iterations: int = 5,
    additional_stop_sequences=None,
) -> AgentExecutor:
    if additional_stop_sequences is None:
        additional_stop_sequences = []

    prompt = BASE_REACT_PROMPT

    prompt = HumanMessagePromptTemplate.from_template(prompt)
    prompt_with_history = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="history"), prompt]
    )

    agent = create_react_agent(
        chat_model,
        tools,
        prompt_with_history,
        output_parser=ReActSingleInputOutputDictParser(),
        stop_sequence=["Observation:"] + additional_stop_sequences,
    )

    agent_executor = AgentExecutor(
        agent=RunnableAgentWithStop(runnable=agent),
        tools=tools,
        verbose=True,
        max_iterations=agent_max_iterations,
        handle_parsing_errors=True,
        early_stopping_method="generate",
        return_intermediate_steps=True,
    )

    return agent_executor
