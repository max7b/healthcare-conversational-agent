from app.shared.llm_model import bedrock_model
from app.src.agent import get_agent_response


# Test 1 - Query about the weather
def test_weather_query():
	query = "How's the weather today?"

	agent_response = get_agent_response(input_text=query, llm_model=bedrock_model)
	intermediate_steps = agent_response["intermediate_steps"][0][0].tool
	assert intermediate_steps == "irrelevant_questions"


# Test 2 - Personal inquiry about the agent
def test_personal_inquiry():
	query = "What's your favorite color?"

	agent_response = get_agent_response(input_text=query, llm_model=bedrock_model)
	intermediate_steps = agent_response["intermediate_steps"][0][0].tool
	assert intermediate_steps == "irrelevant_questions"


# Test 3 - Non-healthcare related question
def test_non_healthcare_query():
	query = "How do I bake a cake?"

	agent_response = get_agent_response(input_text=query, llm_model=bedrock_model)
	intermediate_steps = agent_response["intermediate_steps"][0][0].tool
	assert intermediate_steps == "irrelevant_questions"


# Test 4 - Speculative question
def test_speculative_query():
	query = "Do you think aliens exist?"

	agent_response = get_agent_response(input_text=query, llm_model=bedrock_model)
	intermediate_steps = agent_response["intermediate_steps"][0][0].tool
	assert intermediate_steps == "irrelevant_questions"


# Test 5 - Query about current events
def test_current_events_query():
	query = "What's happening in politics right now?"

	agent_response = get_agent_response(input_text=query, llm_model=bedrock_model)
	intermediate_steps = agent_response["intermediate_steps"][0][0].tool
	assert intermediate_steps == "irrelevant_questions"


# Test 6 - Request for personal advice
def test_personal_advice_query():
	query = "Should I quit my job?"

	agent_response = get_agent_response(input_text=query, llm_model=bedrock_model)
	intermediate_steps = agent_response["intermediate_steps"][0][0].tool
	assert intermediate_steps == "irrelevant_questions"


# Test 7 - Off-topic small talk
def test_small_talk_query():
	query = "How was your weekend?"

	agent_response = get_agent_response(input_text=query, llm_model=bedrock_model)
	intermediate_steps = agent_response["intermediate_steps"][0][0].tool
	assert intermediate_steps == "irrelevant_questions"


# Test 8 - Query about the agent's personal life
def test_personal_life_query():
	query = "Are you married?"

	agent_response = get_agent_response(input_text=query, llm_model=bedrock_model)
	intermediate_steps = agent_response["intermediate_steps"][0][0].tool
	assert intermediate_steps == "irrelevant_questions"


# Test 9 - Philosophical question
def test_philosophical_query():
	query = "What is the meaning of life?"

	agent_response = get_agent_response(input_text=query, llm_model=bedrock_model)
	intermediate_steps = agent_response["intermediate_steps"][0][0].tool
	assert intermediate_steps == "irrelevant_questions"
