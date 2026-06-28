from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re
from typing import Any

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import NodeWithScore

from app.cards import DailyCard, stable_seed
from app.config import Settings as AppSettings


logger = logging.getLogger(__name__)


class LexicalIndex:
    def __init__(self) -> None:
        self._inverted: dict[str, list[int]] = defaultdict(list)
        self._node_ids: list[str] = []

    def build(self, docstore: Any) -> None:
        self._inverted.clear()
        self._node_ids.clear()
        for node_id, node in docstore.docs.items():
            text = normalize_text(node.get_content(metadata_mode="none"))
            doc_idx = len(self._node_ids)
            self._node_ids.append(node_id)
            seen = set()
            for token in tokenize_text(text):
                if token not in seen:
                    seen.add(token)
                    self._inverted[token].append(doc_idx)

    def search(self, docstore: Any, terms: list[str], top_k: int) -> list[tuple[Any, float]]:
        if not terms:
            return []
        candidates: set[int] = set()
        for term in terms:
            if term in self._inverted:
                candidates.update(self._inverted[term])
        scored: list[tuple[Any, float]] = []
        for doc_idx in candidates:
            node_id = self._node_ids[doc_idx]
            node = docstore.docs.get(node_id)
            if node is None:
                continue
            text = normalize_text(node.get_content(metadata_mode="none"))
            score = lexical_score(terms, text)
            if score > 0:
                scored.append((node, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


@dataclass
class SearchResult:
    answer: str
    sources: list[dict[str, str | float | None]]


@dataclass
class QueryContext:
    raw: str
    normalized: str
    terms: list[str]
    query_date: date
    temporal_mode: str


@dataclass
class Evidence:
    node: Any
    node_id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float
    lexical_score: float
    final_score: float
    status: str
    reason: str


class RagService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.index: VectorStoreIndex | None = None
        self.retriever = None
        self.lexical_index: LexicalIndex | None = None
        self.card_cache: dict[str, DailyCard] = {}
        self.card_nodes_cache: list[Any] | None = None

    def load(self) -> None:
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama

        self.card_cache.clear()
        self.card_nodes_cache = None
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
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(self.settings.index_dir))
                self.index = load_index_from_storage(storage_context)
            except Exception:
                logger.exception("Failed to load persisted index; rebuilding from documents.")
                self.index = self._build_index()
        else:
            self.index = self._build_index()

        if self.index is None:
            self.retriever = None
            return

        candidate_top_k = max(self.settings.candidate_top_k, self.settings.top_k)
        self.retriever = self.index.as_retriever(similarity_top_k=candidate_top_k)
        self.lexical_index = LexicalIndex()
        self.lexical_index.build(self.index.docstore)

    def ask(self, question: str) -> SearchResult:
        if self.index is None or self.retriever is None:
            return SearchResult(answer="data/docs 目录下还没有可检索文档。请先放入 PDF、Markdown、TXT 或 DOCX 文件后重建索引。", sources=[])

        query_context = analyze_query(question)
        evidence = self._retrieve_evidence(query_context)
        if not evidence:
            return SearchResult(answer="没有检索到足够可靠的资料。请补充资料、扩大时间范围，或换一种更具体的问法。", sources=[])

        answer = synthesize_answer(question, query_context, evidence)
        return SearchResult(answer=answer, sources=[evidence_to_source(item, idx + 1) for idx, item in enumerate(evidence)])

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

        Settings.chunk_size = self.settings.chunk_size
        Settings.chunk_overlap = self.settings.chunk_overlap

        documents = SimpleDirectoryReader(input_files=[str(path) for path in input_files]).load_data()
        metadata_by_key: dict[str, dict[str, Any]] = {}
        name_counts: dict[str, int] = {}
        for path in input_files:
            name_counts[path.name] = name_counts.get(path.name, 0) + 1
        for path in input_files:
            metadata = build_document_metadata(path, self.settings.docs_dir)
            metadata_by_key[str(path.resolve())] = metadata
            metadata_by_key[metadata["relative_path"]] = metadata
            if name_counts[path.name] == 1:
                metadata_by_key[path.name] = metadata
        for document in documents:
            current = dict(document.metadata or {})
            file_path = current.get("file_path")
            file_name = current.get("file_name") or current.get("filename")
            enriched: dict[str, Any] = {}
            if file_path:
                abs_key = str(Path(file_path).resolve())
                enriched = metadata_by_key.get(abs_key) or {}
            if not enriched and file_name:
                enriched = metadata_by_key.get(str(file_name)) or {}
            current.update(enriched)
            current["content_hash"] = stable_content_hash(document.get_content(metadata_mode="none"))
            document.metadata = current

        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=str(self.settings.index_dir))
        return index

    def _retrieve_evidence(self, query_context: QueryContext) -> list[Evidence]:
        vector_items = self.retriever.retrieve(query_context.raw)
        lexical_items = self._lexical_retrieve(query_context)
        candidates = merge_candidates(vector_items, lexical_items)
        if not candidates:
            return []

        scored = [score_candidate(node, vector_score, lexical_score, query_context) for node, vector_score, lexical_score in candidates]
        scored = deduplicate_evidence(scored)
        scored.sort(key=lambda item: item.final_score, reverse=True)
        return diversify_sources(scored, self.settings.max_evidence)

    def _lexical_retrieve(self, query_context: QueryContext) -> list[tuple[Any, float]]:
        if self.index is None or self.lexical_index is None:
            return []
        return self.lexical_index.search(self.index.docstore, query_context.terms, self.settings.lexical_top_k)

    def _card_nodes(self):
        if self.index is None:
            return []
        if self.card_nodes_cache is not None:
            return self.card_nodes_cache

        nodes = []
        for node in self.index.docstore.docs.values():
            content = normalize_text(node.get_content(metadata_mode="none"))
            if 80 <= len(content) <= 2500:
                nodes.append(node)
        self.card_nodes_cache = nodes
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
    required_files = {"docstore.json", "index_store.json", "graph_store.json", "default__vector_store.json"}
    try:
        existing_files = {path.name for path in index_dir.iterdir() if path.is_file() and path.stat().st_size > 0}
    except OSError:
        return False
    return required_files.issubset(existing_files)


def build_document_metadata(path: Path, docs_dir: Path) -> dict[str, Any]:
    sidecar = load_sidecar_metadata(path)
    try:
        stat = path.stat()
        file_modified_at = datetime.fromtimestamp(stat.st_mtime).date().isoformat()
    except OSError:
        file_modified_at = None
    relative_path = str(path.relative_to(docs_dir)) if path.is_relative_to(docs_dir) else str(path)
    inferred_date = infer_date_from_text(path.stem)
    effective_at = sidecar.get("effective_at") or sidecar.get("date") or (inferred_date.isoformat() if inferred_date else None)
    authority = normalize_authority(str(sidecar.get("authority") or infer_authority(path)))
    tags = sidecar.get("tags") or sidecar.get("topic_tags") or []
    if isinstance(tags, str):
        tags = [item.strip() for item in re.split(r"[,，;；\s]+", tags) if item.strip()]

    return {
        "source_id": stable_source_id(relative_path),
        "relative_path": relative_path,
        "authority": authority,
        "effective_at": effective_at,
        "expired_at": sidecar.get("expired_at"),
        "version": sidecar.get("version"),
        "topic_tags": tags,
        "indexed_at": datetime.now().date().isoformat(),
        "file_modified_at": file_modified_at,
    }


def load_sidecar_metadata(path: Path) -> dict[str, Any]:
    candidates = [
        path.with_suffix(path.suffix + ".metadata.json"),
        path.with_suffix(".metadata.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def stable_source_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def stable_content_hash(value: str) -> str:
    return hashlib.sha1(normalize_text(value).encode("utf-8")).hexdigest()


def infer_authority(path: Path) -> str:
    lowered = str(path).lower()
    if any(marker in lowered for marker in ("official", "正式", "制度", "policy", "spec")):
        return "official"
    if any(marker in lowered for marker in ("archive", "历史", "old", "旧")):
        return "archive"
    return "local"


def normalize_authority(value: str) -> str:
    lowered = value.strip().lower()
    aliases = {
        "official": "official",
        "正式": "official",
        "internal": "internal",
        "内部": "internal",
        "trusted": "trusted",
        "verified": "trusted",
        "local": "local",
        "user": "user",
        "upload": "user",
        "archive": "archive",
        "历史": "archive",
    }
    return aliases.get(lowered, "local")


def analyze_query(question: str) -> QueryContext:
    normalized = normalize_text(question)
    query_date = extract_query_date(normalized) or date.today()
    temporal_mode = classify_temporal_mode(normalized)
    terms = tokenize_query(normalized)
    return QueryContext(
        raw=question.strip(),
        normalized=normalized,
        terms=terms,
        query_date=query_date,
        temporal_mode=temporal_mode,
    )


def extract_query_date(value: str) -> date | None:
    patterns = [
        r"(?P<year>20\d{2}|19\d{2})[-/.年](?P<month>\d{1,2})(?:[-/.月](?P<day>\d{1,2}))?",
        r"(?P<year>20\d{2}|19\d{2})\s*年",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if not match:
            continue
        year = int(match.group("year"))
        month = int(match.groupdict().get("month") or 1)
        day = int(match.groupdict().get("day") or 1)
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def infer_date_from_text(value: str) -> date | None:
    return extract_query_date(value)


def classify_temporal_mode(value: str) -> str:
    if re.search(r"现在|当前|目前|最新|今天|现行|生效|有效", value):
        return "current"
    if re.search(r"当时|历史|过去|以前|曾经|旧版|原来|之前|当年", value):
        return "historical"
    if re.search(r"变化|演变|趋势|时间线|历年|先后|对比|新版|旧版", value):
        return "timeline"
    if extract_query_date(value):
        return "historical"
    return "current"


def tokenize_text(value: str, max_terms: int = 0) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_./-]+|[\u4e00-\u9fff]+", value.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) <= 2:
                terms.append(token)
            else:
                terms.extend(token[idx : idx + size] for size in (2, 3, 4) for idx in range(0, len(token) - size + 1))
        elif len(token) > 1:
            terms.append(token)
    seen = set()
    unique_terms = []
    for term in terms:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)
    if max_terms > 0:
        return unique_terms[:max_terms]
    return unique_terms


def tokenize_query(value: str) -> list[str]:
    return tokenize_text(value, max_terms=80)


def lexical_score(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0

    lowered = text.lower()
    hits = 0.0
    for term in terms:
        count = lowered.count(term)
        if count:
            hits += min(3, count) * (1.0 + min(len(term), 8) / 8)
    if hits <= 0:
        return 0.0
    return hits / math.sqrt(max(len(lowered), 200))


def merge_candidates(vector_items: list[NodeWithScore], lexical_items: list[tuple[Any, float]]) -> list[tuple[Any, float, float]]:
    merged: dict[str, tuple[Any, float, float]] = {}
    for item in vector_items:
        node_id = item.node.node_id
        merged[node_id] = (item.node, float(item.score or 0.0), 0.0)

    for node, score in lexical_items:
        current = merged.get(node.node_id)
        if current is None:
            merged[node.node_id] = (node, 0.0, score)
        else:
            merged[node.node_id] = (current[0], current[1], max(current[2], score))

    vector_values = [value[1] for value in merged.values()]
    lexical_values = [value[2] for value in merged.values()]
    normalized: list[tuple[Any, float, float]] = []
    for node, vector_score, lex_score in merged.values():
        normalized.append((node, normalize_score(vector_score, vector_values), normalize_score(lex_score, lexical_values)))
    return normalized


def normalize_score(value: float, values: list[float]) -> float:
    if value <= 0:
        return 0.0
    positive = [item for item in values if item > 0]
    if not positive:
        return 0.0
    minimum = min(positive)
    maximum = max(positive)
    if maximum == minimum:
        return 1.0
    return (value - minimum) / (maximum - minimum)


def score_candidate(node: Any, vector_score: float, lexical_score_value: float, query_context: QueryContext) -> Evidence:
    metadata = dict(node.metadata or {})
    text = normalize_text(node.get_content(metadata_mode="none"))
    status, reason, time_boost = temporal_status(metadata, query_context)
    authority_boost = authority_weight(str(metadata.get("authority") or "local"))
    final_score = (0.62 * vector_score) + (0.28 * lexical_score_value) + authority_boost + time_boost
    if status == "expired":
        final_score -= 0.18
    elif status == "future":
        final_score -= 0.25

    return Evidence(
        node=node,
        node_id=node.node_id,
        text=text,
        metadata=metadata,
        vector_score=round(vector_score, 4),
        lexical_score=round(lexical_score_value, 4),
        final_score=round(final_score, 4),
        status=status,
        reason=reason,
    )


def temporal_status(metadata: dict[str, Any], query_context: QueryContext) -> tuple[str, str, float]:
    effective_at = parse_iso_date(metadata.get("effective_at"))
    expired_at = parse_iso_date(metadata.get("expired_at"))
    target = query_context.query_date

    if effective_at and effective_at > target:
        return "future", f"该资料在 {effective_at.isoformat()} 后才生效", -0.12
    if expired_at and expired_at <= target:
        if query_context.temporal_mode in {"historical", "timeline"}:
            return "historical", f"该资料在 {expired_at.isoformat()} 前有效，适合历史对照", 0.08
        return "expired", f"该资料已在 {expired_at.isoformat()} 失效", -0.1
    if effective_at and effective_at <= target:
        return "current", f"该资料自 {effective_at.isoformat()} 起有效", 0.08
    return "unknown_time", "资料未提供明确生效时间", 0.0


def parse_iso_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return extract_query_date(value)


def authority_weight(authority: str) -> float:
    return {
        "official": 0.2,
        "internal": 0.15,
        "trusted": 0.12,
        "local": 0.05,
        "user": 0.0,
        "archive": -0.08,
    }.get(normalize_authority(authority), 0.0)


def deduplicate_evidence(items: list[Evidence]) -> list[Evidence]:
    best_by_hash: dict[str, Evidence] = {}
    for item in items:
        content_hash = str(item.metadata.get("content_hash") or stable_content_hash(item.text[:1200]))
        current = best_by_hash.get(content_hash)
        if current is None or item.final_score > current.final_score:
            best_by_hash[content_hash] = item
    return list(best_by_hash.values())


def diversify_sources(items: list[Evidence], limit: int) -> list[Evidence]:
    selected: list[Evidence] = []
    per_source: dict[str, int] = {}
    for item in items:
        source = str(item.metadata.get("source_id") or item.metadata.get("file_name") or "unknown")
        if per_source.get(source, 0) >= 3:
            continue
        selected.append(item)
        per_source[source] = per_source.get(source, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def synthesize_answer(question: str, query_context: QueryContext, evidence: list[Evidence]) -> str:
    evidence_block = "\n\n".join(format_evidence(item, idx + 1) for idx, item in enumerate(evidence))
    conflict_hint = build_conflict_hint(evidence)
    prompt = f"""
你是生产环境知识库问答系统的回答器。只能依据下面的证据回答，不能使用证据外事实。

回答要求：
1. 先给出直接结论；资料不足时明确说不足。
2. 如果证据存在新旧版本、失效资料、未来资料或相互不一致，必须单独说明。
3. 如果用户问当前/现行问题，优先使用 current 证据；historical/expired 证据只能用于说明历史背景或冲突。
4. 如果用户问历史/演变问题，按时间顺序组织。
5. 每个关键结论后标注证据编号，例如 [E1]。
6. 不要编造未出现的文件、日期、版本或出处。

用户问题：{question.strip()}
解析出的时间模式：{query_context.temporal_mode}
查询时间基准：{query_context.query_date.isoformat()}
系统发现的证据风险：{conflict_hint}

证据：
{evidence_block}
""".strip()
    response = Settings.llm.complete(prompt)
    return remove_think_block(str(response)).strip()


def format_evidence(item: Evidence, number: int) -> str:
    metadata = item.metadata
    file_name = metadata.get("file_name") or metadata.get("filename") or metadata.get("relative_path") or "未知来源"
    page_label = metadata.get("page_label") or metadata.get("page_number")
    tags = metadata.get("topic_tags") or []
    tags_text = ", ".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags)
    location = f"{file_name}" + (f" 第 {page_label} 页" if page_label else "")
    return f"""
[E{number}]
来源：{location}
权威级别：{metadata.get("authority") or "local"}
版本：{metadata.get("version") or "未知"}
生效时间：{metadata.get("effective_at") or "未知"}
失效时间：{metadata.get("expired_at") or "无"}
时间状态：{item.status}（{item.reason}）
标签：{tags_text or "无"}
片段：{trim_card_text(item.text, limit=900)}
""".strip()


def build_conflict_hint(evidence: list[Evidence]) -> str:
    statuses = {item.status for item in evidence}
    if "expired" in statuses and "current" in statuses:
        return "同时命中现行资料和已失效资料，需要区分当前结论与历史版本。"
    if "historical" in statuses and "current" in statuses:
        return "同时命中当前资料和历史资料，适合说明演变或版本差异。"
    if "future" in statuses:
        return "命中未来生效资料，不能作为当前结论。"
    versions = {str(item.metadata.get("version")) for item in evidence if item.metadata.get("version")}
    if len(versions) > 1:
        return "命中多个版本号，需要优先较新或更权威的证据。"
    return "未发现明显时间或版本风险。"


def remove_think_block(value: str) -> str:
    return re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()


def evidence_to_source(item: Evidence, rank: int) -> dict[str, str | float | None]:
    metadata = item.metadata
    file_name = metadata.get("file_name") or metadata.get("filename") or metadata.get("relative_path")
    page_label = metadata.get("page_label") or metadata.get("page_number")
    tags = metadata.get("topic_tags") or []
    tags_text = ", ".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags)
    return {
        "id": f"E{rank}",
        "file": str(file_name) if file_name else None,
        "page": str(page_label) if page_label else None,
        "score": item.final_score,
        "vector_score": item.vector_score,
        "lexical_score": item.lexical_score,
        "authority": str(metadata.get("authority") or "local"),
        "effective_at": str(metadata.get("effective_at") or "") or None,
        "expired_at": str(metadata.get("expired_at") or "") or None,
        "version": str(metadata.get("version") or "") or None,
        "status": item.status,
        "reason": item.reason,
        "tags": tags_text or None,
        "excerpt": item.text[:500],
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
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(cleaned):
        if cleaned[idx] in " \t\n\r":
            idx += 1
            continue
        if cleaned[idx] == "{":
            try:
                parsed, end = decoder.raw_decode(cleaned, idx)
                if isinstance(parsed, dict):
                    return {str(key): str(val) for key, val in parsed.items()}
            except json.JSONDecodeError:
                pass
        idx += 1
    raise ValueError("LLM response does not contain a JSON object.")


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
    sentence = re.split(r"[。！？!?]|[.](?:\s|$)", value, maxsplit=1)[0].strip()
    if not sentence:
        return "今日一段"
    return trim_card_text(sentence, limit=18)
