from urllib.parse import quote

import requests
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st_callback = StreamlitCallbackHandler(st.container())


def get_response(query):
    # Make a POST request to the server endpoint
    encoded_query = quote(query)
    response = requests.post(f"http://0.0.0.0:8000/conversation/?query={encoded_query}")
    if response.status_code == 200:
        return response.json()
    else:
        return {"output": "Failed to get response from server."}


st.title("Conversational Agent")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        # Get response
        response = get_response(prompt)
        # Display response
        st.write(response["output"])
