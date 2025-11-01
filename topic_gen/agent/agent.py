import os
import uuid
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from topic_gen.agent.prompt import SYSTEM_PROMPT, TOPIC_GEN_USER_PROMPT

load_dotenv()


@dataclass
class ResponseFormat:
    """Response format for the topic generation agent"""

    # The title of the topic generated (always required)
    topic: str
    # The description of the topic generated (optional)
    description: str | None = None
    # The score of the topic generated (between 0 and 100) (optional)
    score: int | None = None


class TopicGenAgent:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5,
        )
        self.checkpointer = InMemorySaver()
        self.agent = create_agent(
            system_prompt=SYSTEM_PROMPT,
            model=self.model,
            checkpointer=self.checkpointer,
            response_format=ToolStrategy(ResponseFormat),
        )

    def invoke(
        self, tags: list[str], titles: list[str], thread_id: str = uuid.uuid4()
    ) -> ResponseFormat:
        """Invoke the topic generation agent"""
        config = {"configurable": {"thread_id": str(thread_id)}}
        messages = [
            {
                "role": "user",
                "content": TOPIC_GEN_USER_PROMPT(tags=tags, titles=titles),
            },
        ]
        response = self.agent.invoke({"messages": messages}, config=config)
        return response["structured_response"]
