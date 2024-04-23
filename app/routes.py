import uvicorn
from fastapi import FastAPI

from app.shared.llm_model import bedrock_model
from app.src.agent import get_agent_response

app = FastAPI()


@app.post("/conversation/")
async def create_item(query: str):
    agent_response = get_agent_response(input_text=query, llm_model=bedrock_model)
    return agent_response


def start():
    uvicorn.run("app.routes:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
