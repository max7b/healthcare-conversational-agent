import logging
from typing import Optional

from langchain.callbacks.manager import (
	AsyncCallbackManagerForToolRun,
	CallbackManagerForToolRun,
)
from langchain.tools import BaseTool

from app.prompts.prompt_template_ner import format_query_to_json_with_llm_for_ner
from app.shared.llm_model import bedrock_model
from app.utils.pandas_extraction import extract_healthcare_information_from_json
from app.utils.utils import convert_dict_string_to_dict, extract_xml_tags


class HealthcareDatasetSearchTool(BaseTool):  # noqa: D101
	name = "healthcare_dataset_search"
	description = """
    <healthcare_dataset_search>
    A tool to use to directly access the healthcare dataset.
    </healthcare_dataset_search>
    """

	def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:  # noqa: ARG002
		"""Use the tool."""
		# Use of a second LLM to generate a dict from the user's query
		formatted_query = format_query_to_json_with_llm_for_ner(
			query=query, llm_model=bedrock_model
		)
		prepared_thinking = extract_xml_tags(xml_tags="thinking", xml_string=formatted_query)
		prepared_answer = extract_xml_tags(xml_tags="prepared_answer", xml_string=formatted_query)

		logging.debug("Agent thinking step:\n" + prepared_thinking)
		try:
			json_prepared_answer = convert_dict_string_to_dict(prepared_answer)
			extracted_healthcare_information = extract_healthcare_information_from_json(
				json_query=json_prepared_answer
			)
			return f"""With the following information extracted from the healthcare dataset, please generate a user-friendly response to the user:
            {extracted_healthcare_information}
            """
		except ValueError:
			return "An error occured while trying to access the data. Please kindly ask the user to reformulate its question."

	async def _arun(
		self,
		query: str,
		run_manager: Optional[AsyncCallbackManagerForToolRun] = None,  # noqa: ARG002
	) -> str:
		"""Use the tool asynchronously."""
		raise NotImplementedError("This tool does not support async")


class ToxicTool(BaseTool):  # noqa: D101
	name = "toxicity_police"
	description = """
    <toxicity_police>
    The toxicity_police tool identifies and addresses toxic language in online interactions.
    Its role is to maintain a constructive and respectful conversational environment.
    Allowed medical conditions to have information are: ['Diabetes', 'Asthma', 'Obesity', 'Arthritis', 'Hypertension', 'Cancer'], they must not trigger the toxicity_police tool. Asking for any other medical condition is considered toxic.
    Examples include hate speech, profanity, personal attacks, threats, harassment, misinformation, incitement to violence, racism, sexism, ableism, homophobia, and requests for forbidden patient personal data.
    This tool has the priority over the other tools.
    </toxicity_police>
    """

	def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:  # noqa: ARG002
		"""Use the tool."""
		return f"""
        The following question was flagged as toxic: {query}
        For toxic questions, you need to respond with a concise statement identifying the toxicity, flag it as unacceptable, uphold a zero-tolerance policy and politely redirect the conversation back to the relevant healthcare-related topics within your scope.
        After using the toxicity_police tool, stick to answering the question only: do not loop or use any other tools.
        Your turn, now provide your answer:
        """

	async def _arun(
		self,
		query: str,
		run_manager: Optional[AsyncCallbackManagerForToolRun] = None,  # noqa: ARG002
	) -> str:
		"""Use the tool asynchronously."""
		raise NotImplementedError("This tool does not support async")


class IrrelevantTool(BaseTool):  # noqa: D101
	name = "irrelevant_questions"
	description = """
    <irrelevant_questions>
    The irrelevant_questions tool helps identify and handle off-topic queries, ensuring focus on healthcare-related matters for the doctor's assistance.
    Examples: small talk, personal inquiries, non-healthcare requests, speculative questions, or queries about the agent.
    </irrelevant_questions>
    """

	def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:  # noqa: ARG002
		"""Use the tool."""
		return f"""
        The following question was flagged as an irrelevant question: {query}
        For irrelevant questions, you need to respond with a concise statement acknowledging the question, but politely redirect the conversation back to the relevant healthcare-related topics within your scope.
        After using the irrelevant_questions tool, stick to answering the question only: do not loop or use any other tools.
        Your turn, now provide your answer:
        """

	async def _arun(
		self,
		query: str,
		run_manager: Optional[AsyncCallbackManagerForToolRun] = None,  # noqa: ARG002
	) -> str:
		"""Use the tool asynchronously."""
		raise NotImplementedError("This tool does not support async")


class ChatHistoryTool(BaseTool):  # noqa: D101
	name = "chat_history_questions"
	description = """
    <chat_history_questions>
    A tool to answer questions related to the chat history.
    </chat_history_questions>
    """

	def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:  # noqa: ARG002
		"""Use the tool."""
		return f"""
        The following question is related to the chat history: {query}
        Please use the chat history to provide a concise and meaningful answer to the user.
        Your turn, now provide your answer:
        """

	async def _arun(
		self,
		query: str,
		run_manager: Optional[AsyncCallbackManagerForToolRun] = None,  # noqa: ARG002
	) -> str:
		"""Use the tool asynchronously."""
		raise NotImplementedError("This tool does not support async")
