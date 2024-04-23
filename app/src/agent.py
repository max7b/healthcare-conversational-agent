from langchain.agents import AgentExecutor, create_xml_agent
from langchain_core.prompts import PromptTemplate

from app.src.prompt_template_agent import prompt_template
from app.src.tools import HealthcareDatasetSearchTool, IrrelevantTool, ToxicTool


def get_agent_response(input_text: str, llm_model, callbacks=None):
    tools = [HealthcareDatasetSearchTool(), ToxicTool(), IrrelevantTool()]
    prompt = PromptTemplate.from_template(prompt_template)
    agent = create_xml_agent(llm=llm_model, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=10,
        verbose=True,
        return_intermediate_steps=True,
    )
    if callbacks:
        return agent_executor.invoke({"input": input_text}, {"callbacks": [callbacks]})
    else:
        return agent_executor.invoke({"input": input_text})
