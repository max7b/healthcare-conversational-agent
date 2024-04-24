import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from app.shared.llm_model import bedrock_model
from app.src.agent import get_agent_response

st_callback = StreamlitCallbackHandler(st.container())
st.title("Conversational Agent")

# initialize the chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Re-render the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# The agent starts by engaging conversation
if not st.session_state.chat_history:
    with st.chat_message("assistant"):
        first_message = "Hello, how can I assist you today?"
        st.markdown(first_message)
        st.session_state.chat_history.append(
            {"role": "assistant", "text": first_message}
        )

if prompt := st.chat_input():
    # append the user's latest message to the chat history
    st.session_state.chat_history.append({"role": "user", "text": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        # Get response
        response = get_agent_response(
            input_text=prompt,
            llm_model=bedrock_model,
            session_id="1234",
            callbacks=st_callback,
        )
        # Display response
        st.write(response["output"])

    # append the bot's latest message to the chat history
    st.session_state.chat_history.append(
        {"role": "assistant", "text": response["output"]}
    )
