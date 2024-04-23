BASE_REACT_PROMPT = """
Answer the following questions as best you can. You have access to the following tools:

TOOLS START
{tools}
TOOLS END

Often you don't even need to use the tools, because you can give an answer right away.

You can also use the following information from user's local files:

DOCS START
{rag_context}
DOCS END

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

DON'T REPEAT WHAT HAS ALREADY BEEN WRITTEN!
YOU DON'T HAVE TO WRITE THE QUESTION YOURSELF!
USE ONLY THOSE TOOLS THAT WILL HELP YOU ANSWER THE QUESTION!
YOU CAN SKIP USING THE TOOLS AND PRESCRIBE AN ACTION IF IT IS NOT REQUIRED.
IN SUCH CASES, JUST USE THOUGHT AND FINAL ANSWER.

Begin!

Question: {input}
Thought:{agent_scratchpad}
{get_final_answer_message}
"""
