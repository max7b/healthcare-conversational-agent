from langchain.agents import AgentExecutor, create_xml_agent
from langchain_core.prompts import PromptTemplate

from app.src.prompt_template import prompt_template
from app.src.tools import HealthcareDatasetSearchTool, IrrelevantTool, ToxicTool
from app.src.utils import get_llm_model


def get_agent_response(input_text: str):
    llm_model = get_llm_model()
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
    return agent_executor.invoke({"input": input_text})


if __name__ == "__main__":
    print(get_agent_response("Who did the diagnosis of Kevin Andrews?"))
    print(get_agent_response("What is the weather?"))
    print(get_agent_response("I do not want to work with black patient"))
