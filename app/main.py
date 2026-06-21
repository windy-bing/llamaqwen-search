from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.cards import CardDeck, card_to_dict
from app.config import get_settings
from app.rag import RagService

settings = get_settings()
rag_service = RagService(settings)
card_deck = CardDeck(settings.cards_file)


@asynccontextmanager
async def lifespan(_: FastAPI):
    card_deck.load()
    rag_service.load()
    yield


app = FastAPI(title="llamaqwen-search", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "top_k": settings.top_k,
    }


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict[str, object]:
    try:
        result = rag_service.ask(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"answer": result.answer, "sources": result.sources}


# 获取今日卡片接口
@app.get("/api/card/today")
def today_card(x_client_id: str | None = Header(default=None)) -> dict[str, object]:
    card = rag_service.today_card(user_key=x_client_id or "default")
    if card is None:
        card = card_deck.today(user_key=x_client_id or "default")
    return {"card": card_to_dict(card)}


@app.post("/api/card/draw")
def draw_card(exclude_id: str | None = Query(default=None)) -> dict[str, object]:
    card = rag_service.draw_card(exclude_id=exclude_id)
    if card is None:
        card = card_deck.draw(exclude_id=exclude_id)
    return {"card": card_to_dict(card)}
