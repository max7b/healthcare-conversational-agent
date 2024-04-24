from langchain.agents import AgentExecutor, create_xml_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.prompts.prompt_template_agent import prompt_template
from app.src.tools import (
    ChatHistoryTool,
    HealthcareDatasetSearchTool,
    IrrelevantTool,
    ToxicTool,
)

# Notre DB simple pour gérer l'historique de conversation
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_agent_response(input_text: str, llm_model, session_id="1234", callbacks=None):
    tools = [
        HealthcareDatasetSearchTool(),
        ToxicTool(),
        IrrelevantTool(),
        ChatHistoryTool(),
    ]
    prompt = PromptTemplate.from_template(prompt_template)
    agent = create_xml_agent(llm=llm_model, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=10,
        verbose=True,
        return_intermediate_steps=True,
    )
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_chat_history.invoke(
        {"input": input_text},
        config={
            "configurable": {"session_id": session_id},
            "callbacks": [callbacks] if callbacks else [],
        },
    )
