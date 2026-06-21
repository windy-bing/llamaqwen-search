from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import random
import re

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore

from app.cards import DailyCard, stable_seed
from app.config import Settings as AppSettings


@dataclass
class SearchResult:
    answer: str
    sources: list[dict[str, str | float | None]]


class RagService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.index: VectorStoreIndex | None = None
        self.query_engine: RetrieverQueryEngine | None = None
        self.card_cache: dict[str, DailyCard] = {}

    def load(self) -> None:
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama

        self.settings.docs_dir.mkdir(parents=True, exist_ok=True)
        self.settings.index_dir.mkdir(parents=True, exist_ok=True)

        Settings.llm = Ollama(
            model=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
            request_timeout=120.0,
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
        )

        has_existing_index = has_persisted_index(self.settings.index_dir)
        if has_existing_index and not self.settings.rebuild_index:
            storage_context = StorageContext.from_defaults(persist_dir=str(self.settings.index_dir))
            self.index = load_index_from_storage(storage_context)
        else:
            self.index = self._build_index()

        if self.index is None:
            self.query_engine = None
            return

        retriever = self.index.as_retriever(similarity_top_k=self.settings.top_k)
        synthesizer = get_response_synthesizer(response_mode="compact")
        self.query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)

    def ask(self, question: str) -> SearchResult:
        if self.query_engine is None:
            return SearchResult(answer="data/docs 目录下还没有可检索文档。请先放入 PDF、Markdown、TXT 或 DOCX 文件后重建索引。", sources=[])

        response = self.query_engine.query(
            "请基于检索到的资料用中文回答。资料不足时明确说明不知道，不要编造。\n\n问题："
            + question.strip()
        )
        source_nodes = getattr(response, "source_nodes", []) or []
        return SearchResult(answer=str(response), sources=[node_to_source(item) for item in source_nodes])

    def has_document_cards(self) -> bool:
        return bool(self._card_nodes())

    def today_card(self, user_key: str = "default", today: date | None = None) -> DailyCard | None:
        nodes = self._card_nodes()
        if not nodes:
            return None

        current_date = today or date.today()
        seed = stable_seed(f"{current_date.isoformat()}:{user_key}")
        return self._generate_card(nodes[seed % len(nodes)])

    def draw_card(self, exclude_id: str | None = None) -> DailyCard | None:
        nodes = self._card_nodes()
        if not nodes:
            return None

        candidates = [node for node in nodes if node.node_id != exclude_id]
        return self._generate_card(random.choice(candidates or nodes))

    def _build_index(self) -> VectorStoreIndex | None:
        input_files = list_supported_files(self.settings.docs_dir)
        if not input_files:
            return None

        documents = SimpleDirectoryReader(input_files=[str(path) for path in input_files]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=str(self.settings.index_dir))
        return index

    def _card_nodes(self):
        if self.index is None:
            return []

        nodes = []
        for node in self.index.docstore.docs.values():
            content = normalize_text(node.get_content(metadata_mode="none"))
            if 80 <= len(content) <= 2500:
                nodes.append(node)
        return nodes

    def _generate_card(self, node) -> DailyCard:
        cached = self.card_cache.get(node.node_id)
        if cached is not None:
            return cached

        fallback = node_to_card(node)
        try:
            card = generate_share_card(node, fallback)
        except Exception:
            card = fallback

        self.card_cache[node.node_id] = card
        return card


def list_supported_files(docs_dir: Path) -> list[Path]:
    suffixes = {".pdf", ".txt", ".md", ".docx"}
    return sorted(path for path in docs_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def has_persisted_index(index_dir: Path) -> bool:
    return any(path.is_file() and path.name != ".gitkeep" for path in index_dir.iterdir())


def node_to_source(item: NodeWithScore) -> dict[str, str | float | None]:
    metadata = dict(item.node.metadata or {})
    file_name = metadata.get("file_name") or metadata.get("filename")
    page_label = metadata.get("page_label") or metadata.get("page_number")
    return {
        "file": str(file_name) if file_name else None,
        "page": str(page_label) if page_label else None,
        "score": item.score,
        "excerpt": item.node.get_content(metadata_mode="none")[:500],
    }


def node_to_card(node) -> DailyCard:
    metadata = dict(node.metadata or {})
    content = normalize_text(node.get_content(metadata_mode="none"))
    file_name = metadata.get("file_name") or metadata.get("filename") or "向量资料"
    page_label = metadata.get("page_label") or metadata.get("page_number")
    reference = f"第 {page_label} 页" if page_label else "索引片段"
    title = build_card_title(content)

    return DailyCard(
        id=node.node_id,
        title=title,
        text=trim_card_text(content),
        source=str(file_name),
        reference=reference,
        action="把这一段和今天要处理的问题联系起来，先写下一条可执行动作。",
    )


def generate_share_card(node, fallback: DailyCard) -> DailyCard:
    content = normalize_text(node.get_content(metadata_mode="none"))
    prompt = build_card_prompt(content, fallback.source, fallback.reference)
    response = Settings.llm.complete(prompt)
    payload = parse_json_object(str(response))

    title = clean_generated_text(payload.get("title", ""), limit=22)
    text = clean_generated_text(payload.get("text", ""), limit=110)
    action = clean_generated_text(payload.get("action", ""), limit=50)

    if len(title) < 2 or len(text) < 18:
        return fallback

    return DailyCard(
        id=fallback.id,
        title=title,
        text=text,
        source=fallback.source,
        reference=fallback.reference,
        action=action or fallback.action,
    )


def build_card_prompt(content: str, source: str, reference: str) -> str:
    return f"""
你要把资料片段改写成一张适合分享的中文“每日思考卡”。

要求：
1. 不要逐字摘抄长段原文，要提炼成一句有启发的摘意。
2. 观点必须能从资料片段中得到支撑，不要编造资料外事实。
3. 语言要像给普通读者看的卡片：清楚、短、有行动感。
4. title 控制在 4-12 个中文字符。
5. text 控制在 40-90 个中文字符。
6. action 控制在 12-28 个中文字符。
7. 只输出 JSON，不要 Markdown，不要解释。

JSON 格式：
{{"title":"", "text":"", "action":""}}

来源：{source}
依据：{reference}
资料片段：
{trim_card_text(content, limit=1200)}
""".strip()


def parse_json_object(value: str) -> dict[str, str]:
    cleaned = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response does not contain a JSON object.")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object.")
    return {str(key): str(val) for key, val in parsed.items()}


def clean_generated_text(value: str, limit: int) -> str:
    cleaned = normalize_text(value).strip("「」\"'` ")
    return trim_card_text(cleaned, limit=limit)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def trim_card_text(value: str, limit: int = 180) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip(" ，。；、") + "..."


def build_card_title(value: str) -> str:
    sentence = re.split(r"[。！？.!?]", value, maxsplit=1)[0].strip()
    if not sentence:
        return "今日一段"
    return trim_card_text(sentence, limit=18)
