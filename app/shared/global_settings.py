"""Global Settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Define BASEDIR for the Project
BASEDIR = Path(__file__).parents[2]

# Load environment variables locally
if os.path.isfile(os.path.join(BASEDIR, ".env")):
	load_dotenv(os.path.join(BASEDIR, ".env"))

# -- API Keys
AWS_CREDENTIALS_PROFILE_NAME = os.environ.get("AWS_CREDENTIALS_PROFILE_NAME", None)

# -- LLM settings
AWS_REGION_NAME = "us-east-1"
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

LLM_TOP_K = 1  # Number of top tokens to consider during text generation (1 for most likely token)
LLM_TOP_P = (
	0.01  # Threshold probability for token selection (low value for reproducible generation)
)
LLM_TEMPERATURE = 0  # Controls randomness during text generation (0 for most probable token)
