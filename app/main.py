from __future__ import annotations

from fastapi import FastAPI

from .engine import handle_chat
from .schemas import ChatRequest, ChatResponse


app = FastAPI(title="SHL Assessment Agent", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    reply, recommendations, end_of_conversation = handle_chat([message.model_dump() for message in request.messages])
    return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=end_of_conversation)
