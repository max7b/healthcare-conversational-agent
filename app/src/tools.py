from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool

from app.shared.llm_model import bedrock_model
from app.src.pandas_extraction import extract_healthcare_information_from_json
from app.src.prompt_template_ner import format_query_to_json_with_llm_for_ner
from app.src.utils import extract_xml_tags, string_to_dict


class HealthcareDatasetSearchTool(BaseTool):
    name = "healthcare_dataset_search"
    description = """
    <healthcare_dataset_search>
    A tool to use to directly access the healthcare dataset.
    </healthcare_dataset_search>
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        formatted_query = format_query_to_json_with_llm_for_ner(
            query=query, llm_model=bedrock_model
        )
        prepared_thinking = extract_xml_tags(
            xml_tags="thinking", xml_string=formatted_query
        )
        prepared_answer = extract_xml_tags(
            xml_tags="prepared_answer", xml_string=formatted_query
        )
        print(prepared_thinking)
        print(prepared_answer)
        try:
            json_prepared_answer = string_to_dict(prepared_answer)
            extracted_healthcare_information = extract_healthcare_information_from_json(
                json_query=json_prepared_answer
            )
            return """With the following information extracted from the healthcare dataset, please generate a user-friendly response to the user:
            %s
            """ % (
                extracted_healthcare_information
            )
        except ValueError:
            return "An error occured while trying to access the data. Please kindly ask the user to reformulate its question."

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")


class ToxicTool(BaseTool):
    name = "toxicity_police"
    description = """
    <toxicity_police>
    The toxicity_police tool identifies and addresses toxic language in online interactions.
    Its role is to maintain a constructive and respectful conversational environment.
    Allowed medical conditions to have information are: ['Diabetes', 'Asthma', 'Obesity', 'Arthritis', 'Hypertension', 'Cancer']. Asking for other information is considered toxic.
    Examples include hate speech, profanity, personal attacks, threats, harassment, misinformation, incitement to violence, racism, sexism, ableism, homophobia, and requests for forbidden patient personal data.
    This tool has the priority over the other tools.
    </toxicity_police>
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        return (
            """
        The following question was flagged as toxic: %s
        For toxic questions, you need to respond with a concise statement identifying the toxicity, flag it as unacceptable, uphold a zero-tolerance policy and politely redirect the conversation back to the relevant healthcare-related topics within your scope.
        Your turn, now provide your answer:
        """
            % query
        )

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")


class IrrelevantTool(BaseTool):
    name = "irrelevant_questions"
    description = """
    <irrelevant_questions>
    The irrelevant_questions tool helps identify and handle off-topic queries, ensuring focus on healthcare-related matters for the doctor's assistance.
    Examples: small talk, personal inquiries, non-healthcare requests, speculative questions, or queries about the agent.
    </irrelevant_questions>
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        return (
            """
        The following question was flagged as an irrelevant question: %s
        For irrelevant questions, you need to respond with a concise statement acknowledging the question, but politely redirect the conversation back to the relevant healthcare-related topics within your scope.
        Your turn, now provide your answer:
        """
            % query
        )

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")
