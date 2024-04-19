from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool

from app.src.llm_ner_to_json import format_query_to_json_with_llm_for_ner
from app.src.utils import extract_xml_tags


class HealthcareDatasetSearchTool(BaseTool):
    name = "healthcare_dataset_search"
    description = """
    <healthcare_dataset_search>
    A tool to use to directly access the healthcare dataset.
    </healthcare_dataset_search>
    """

    def __init__(self):
        super().__init__()
        # Return Direct ends the chain after using the tool.
        self.return_direct = True

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        formatted_query = format_query_to_json_with_llm_for_ner(query=query)
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

    def __init__(self):
        super().__init__()
        # Return Direct ends the chain after using the tool.
        self.return_direct = True

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
    <irrelevant_questions>
    The irrelevant_questions tool helps identify and handle off-topic queries, ensuring focus on healthcare-related matters for the doctor's assistance.
    Examples: small talk, personal inquiries, non-healthcare requests, speculative questions, or queries about the agent.
    </irrelevant_questions>
    """

    def __init__(self):
        super().__init__()
        # Return Direct ends the chain after using the tool.
        self.return_direct = True

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
