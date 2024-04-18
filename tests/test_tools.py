from app.src.agent import get_agent_response


def test_custom_search_tool():
    query = "What is the energy consumption?"

    agent_response = get_agent_response(input_text=query)
    intermediate_steps = agent_response["intermediate_steps"][0][0].tool
    assert intermediate_steps == "search_energy_consumption"
