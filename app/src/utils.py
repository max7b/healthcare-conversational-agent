import ast
import re

from langchain_community.chat_models import BedrockChat

from app.shared.global_settings import (
    AWS_CREDENTIALS_PROFILE_NAME,
    AWS_REGION_NAME,
    BEDROCK_MODEL_ID,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
)


def get_llm_model():
    model_kwargs = {
        "top_k": LLM_TOP_K,
        "top_p": LLM_TOP_P,
        "temperature": LLM_TEMPERATURE,
    }
    model = BedrockChat(
        credentials_profile_name=AWS_CREDENTIALS_PROFILE_NAME,
        region_name=AWS_REGION_NAME,
        model_id=BEDROCK_MODEL_ID,
        streaming=False,
        model_kwargs=model_kwargs,
    )
    return model


def extract_xml_tags(xml_tags: str, xml_string: str) -> str:
    """Extracts the xml string from tag using regular expression pattern.

    Args:
         xml_tags(str): The xml element to match.
         xml_string(str): The string to match against.
    Returns:
         The first match.

    """
    pattern = f"<{xml_tags}.*?>(.*?)</{xml_tags}>"
    # Find all matches of the pattern in the XML string
    matches = re.findall(pattern, xml_string, re.DOTALL)
    return matches[0].replace("\n", "") if matches else ""


def string_to_dict(input_str):
    """
    Convert a string representation of a dictionary into a Python dictionary.

    Parameters:
        input_str (str): The string representation of the dictionary.

    Returns:
        dict: The dictionary object.
    """
    try:
        return ast.literal_eval(input_str)
    except SyntaxError:
        raise ValueError("Input string is not a valid dictionary")
