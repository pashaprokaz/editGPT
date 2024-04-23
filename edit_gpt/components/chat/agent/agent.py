from typing import Any, List, Tuple

from langchain.agents.agent import RunnableAgent
from langchain_core.agents import AgentAction, AgentFinish

GET_FINAL_ANSWER_MESSAGE = (
    "\nFrom now on you should only write thought and final answer! NOTHING ELSE! You can't use action anymore! "
    "Use only thought and final answer\n"
)


class RunnableAgentWithStop(RunnableAgent):
    observation_prefix: str = "Observation: "
    llm_prefix: str = "Thought: "

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."}, ""
            )
        elif early_stopping_method == "generate":
            new_inputs = {
                "intermediate_steps": intermediate_steps,
                "get_final_answer_message": GET_FINAL_ANSWER_MESSAGE,
            }
            full_inputs = {**kwargs, **new_inputs}
            output = self.runnable.invoke(full_inputs)
            return output
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, "
                f"got {early_stopping_method}"
            )
