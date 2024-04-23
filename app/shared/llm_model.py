from langchain_community.chat_models import BedrockChat

from app.shared.global_settings import (
    AWS_CREDENTIALS_PROFILE_NAME,
    AWS_REGION_NAME,
    BEDROCK_MODEL_ID,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
)

model_kwargs = {
    "top_k": LLM_TOP_K,
    "top_p": LLM_TOP_P,
    "temperature": LLM_TEMPERATURE,
}

bedrock_model = BedrockChat(
    credentials_profile_name=AWS_CREDENTIALS_PROFILE_NAME,
    region_name=AWS_REGION_NAME,
    model_id=BEDROCK_MODEL_ID,
    streaming=False,
    model_kwargs=model_kwargs,
)
