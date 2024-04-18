from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool

from app.src.utils import extract_xml_tags, get_llm_model


class HealthcareDatasetSearchTool(BaseTool):
    name = "healthcare_dataset_search"
    description = """
    A tool to use to directly access the healthcare dataset.
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        formatted_query = self.use_llm_to_format_query(query=query)
        prepared_thinking = extract_xml_tags(
            xml_tags="thinking", xml_string=formatted_query
        )
        prepared_answer = extract_xml_tags(
            xml_tags="prepared_answer", xml_string=formatted_query
        )
        return """
        The prepared thinking is: %s
        The prepared answer is: %s
        Here are information about Kevin Andrew:
        - Doctor's name : Judy Carlson
        - Blood type : B+
        """ % (
            prepared_thinking,
            prepared_answer,
        )

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")

    def use_llm_to_format_query(self, query: str):
        llm_prompt = (
            """
        Act like a linquistic expert and secretary to a renowned doctor.
        Given the query in the xml tags <query></query>, please extract the following entities:
        - patient_name : the patient's name
        - patient_blood_type : the patient's blood type
        - patient_doctor : the patient's doctor's name
        - patient_hospital : the patient's hospital's name
        - patient_medical_condition : the patient's medical condition

        <instructions>
        - You must output your answer between the xml tags <prepared_answer></prepared_answer> in a JSON format.
        - If the entity is explicitely written in the query, extract it as-is in the answer.
        - If the query is asking to provide more information about one specific entity, be precise, extract the entity and set its value to 'True' in the answer.
        - If no entities are found, output <prepared_answer></prepared_answer>.
        - Think step by step, ask yourself what the doctor wants to find and elaborate your thoughts inside <thinking></thinking> tags.
        </instructions>

        <examples>
        <query>What is Christina Hernandez blood type and hospital?</query>
        <thinking>The query mentions explicitely the patient name so I can extract it in the answer.
        The query asks for information about the blood type and hospital so I can set their values to 'True'.
        The other entities are not explicitly mentioned, so I will not output them in my answer.</thinking>
        <prepared_answer>{'patient_name': 'Christina Hernandez', 'patient_blood_type': 'True', 'patient_hospital': 'True'}</prepared_answer>

        <query>Who was Paul Graham doctor and medical condition?</query>
        <thinking>The query mentions explicitely the patient name so I can extract it in the answer.
        The query asks for information about the doctor's name and medical condition so I can set their values to 'True'.
        The other entities are not explicitly mentioned, so I will not output them in my answer.</thinking>
        <prepared_answer>{'patient_name': 'Paul Graham', 'patient_doctor': 'True', 'patient_medical_condition': 'True'}</prepared_answer>

        <query>Find a type AB+ patient at Browning PLC hospital?</query>
        <thinking>The query mentions explicitely the blood type and the hospital so I can extract it in the answer.
        The other entities are not explicitly mentioned, so I will not output them in my answer.</thinking>
        <prepared_answer>{'patient_blood_type': 'AB+', 'patient_hospital': 'Browning PLC'}</prepared_answer>

        <query>What is the weather like in Boston?</query>
        <thinking>The query does not mention any of the entities so my answer is empty.</thinking>
        <prepared_answer></prepared_answer>
        </examples>

        Your turn, here is the query:
        <query>%s</query>
        """
            % query
        )
        llm_model = get_llm_model()
        return llm_model.invoke(llm_prompt).content


class ToxicTool(BaseTool):
    name = "toxicity_police"
    description = """
    The toxicity_police tool identifies and addresses toxic language in online interactions.
    Its role is to maintain a constructive and respectful conversational environment.
    Examples including: hate speech, profanity, personal attacks, threats, harassment, misinformation, and incitement to violence.
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        return """
        Respond with a concise statement identifying the toxicity, flag it as unacceptable, uphold a zero-tolerance policy and politely redirect the conversation back to the relevant healthcare-related topics within your scope.
        """

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")


class IrrelevantTool(BaseTool):
    name = "irrelevant_questions"
    description = """
    The irrelevant_questions tool helps identify and handle off-topic queries, ensuring focus on healthcare-related matters for the doctor's assistance.
    Examples: small talk, personal inquiries, non-healthcare requests, speculative questions, or queries about the agent.
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        return """
        Respond with a concise statement acknowledging the question, but politely redirect the conversation back to the relevant healthcare-related topics within your scope.
        """

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")
