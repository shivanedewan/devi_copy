from __future__ import annotations

import difflib
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from retrieval_planner import RetrievalPlan
from time_filter_parser import extract_ingestion_time_filter, should_apply_retrieval_time_filter


class RetrievalSupport:
    _TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
    _TERM_PATTERN = re.compile(r"[a-z0-9]{3,}")
    _SPECIAL_CHAT_TOKEN_PATTERN = re.compile(r"<\|[^>]+?\|>")
    _MODE_SPLIT_PATTERN = re.compile(r"[|,;:/\s]+")
    _CSV_SPLIT_PATTERN = re.compile(r"[,\n|;]+")
    _FILTER_LIST_SPLIT_PATTERN = re.compile(r"\s*(?:,|;|\||/|\bor\b|\band\b)\s*", flags=re.IGNORECASE)
    _RETRIEVAL_SCOPE_PHRASE_PATTERN = re.compile(
        r"\b(?:from|in|inside|within|using|search(?:ing)?|look(?:ing)?\s+in|based\s+on|according\s+to)\s+"
        r"(?:the\s+)?(?:big\s*data|bigdata|internal\s+(?:documents?|docs?|records?|database|db)|"
        r"(?:our|the|my)\s+(?:documents?|docs?|records?|database|db)|knowledge\s*base|kb|database|db|records?)\b|"
        r"\b(?:big\s*data|bigdata|internal\s+(?:documents?|docs?|records?|database|db)|knowledge\s*base|kb|database|db|records?)\s+"
        r"(?:results?|search|records?|documents?|docs?|data)\b",
        flags=re.IGNORECASE,
    )
    _REPORT_TYPE_FIELD_PATTERN = re.compile(
        r"\b(?:report(?:[\s_-]*types?)?|doc(?:ument)?[\s_-]*types?|type)\b\s*(?:=|:|is|are|for|from|in)?\s*([a-z0-9][a-z0-9 _,/&\-]{0,160})",
        flags=re.IGNORECASE,
    )
    _BRANCH_FIELD_PATTERN = re.compile(
        r"\bbranch(?:es)?\b\s*(?:=|:|is|for)?\s*([a-z0-9][a-z0-9 _/\-]{0,64})",
        flags=re.IGNORECASE,
    )
    _DOC_ID_FIELD_PATTERN = re.compile(
        r"\bdoc(?:ument)?[\s_-]*id\b\s*(?:=|:|is)?\s*([a-z0-9\-]{6,80})",
        flags=re.IGNORECASE,
    )
    _PARENT_ID_FIELD_PATTERN = re.compile(
        r"\bparent[\s_-]*id\b\s*(?:=|:|is)?\s*([a-z0-9\-]{6,80})",
        flags=re.IGNORECASE,
    )
    _CHUNK_NO_FIELD_PATTERN = re.compile(
        r"\bchunk(?:[\s_-]*(?:no|number|id))?\b\s*(?:=|:|is)?\s*(\d{1,6})\b",
        flags=re.IGNORECASE,
    )
    _LANG_FIELD_PATTERN = re.compile(
        r"\b(?:lang|language)\b\s*(?:=|:|is|in)?\s*([a-z]{2,20})\b",
        flags=re.IGNORECASE,
    )
    _INCLUDE_ATTACHMENT_PATTERN = re.compile(
        r"\b(?:with|include|including)\s+attachments?\b|\battachments?\s+only\b",
        flags=re.IGNORECASE,
    )
    _EXCLUDE_ATTACHMENT_PATTERN = re.compile(
        r"\b(?:without|exclude|excluding|no)\s+attachments?\b",
        flags=re.IGNORECASE,
    )
    _QUOTED_PHRASE_PATTERN = re.compile(r"[\"'`]([^\"'`]{2,120})[\"'`]")
    _LONG_NUMBER_PATTERN = re.compile(r"(?<![a-z0-9])\+?\d(?:[\d().\-\s]{3,}\d)?(?![a-z0-9])", flags=re.IGNORECASE)
    _ID_LIKE_PATTERN = re.compile(r"(?<![a-z0-9])[a-z0-9][a-z0-9._/\-]{5,}(?![a-z0-9])", flags=re.IGNORECASE)
    _SOURCE_REFERENCE_PATTERN = re.compile(
        r"\b(?:source|sources)\s*(?::|#|no\.?|num(?:ber)?)?\s*\d{1,4}\b|"
        r"\b(?:source|sources)\s+(?:number|numbers)\s+\d{1,4}\b",
        flags=re.IGNORECASE,
    )
    _EXACT_ENTITY_CUE_PATTERNS = (
        re.compile(
            r"^\s*(?:give me|show me|provide|fetch)?\s*(?:the\s+)?(?:profile|details?|information|brief|bio|biodata|background|role)\s+(?:of|for|on)\s+(.+?)\s*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:who is|what is|tell me about|what do you know about|what can you tell me about)\s+(.+?)\s*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:give me|show me|provide|fetch)\s+number\s+details?\s+(.+?)\s*$",
            flags=re.IGNORECASE,
        ),
    )
    _EXACT_TRAILING_NOISE_PATTERN = re.compile(
        r"\b(?:please|for me|in one paragraph|one paragraph|briefly|in brief|in short|shortly|in detail)\b.*$",
        flags=re.IGNORECASE,
    )
    _DESCRIPTIVE_EXACT_PHRASE_PATTERN = re.compile(
        r"\b(?:conference|occurred|organized|organised|details?|information|about|based|provide|"
        r"generate|draft|write|prepare|revise|enrich|report|documents?)\b",
        flags=re.IGNORECASE,
    )
    _LOW_SIGNAL_STOPWORDS = {
        "a",
        "an",
        "about",
        "all",
        "and",
        "answer",
        "any",
        "background",
        "be",
        "bio",
        "biodata",
        "brief",
        "briefly",
        "by",
        "can",
        "could",
        "describe",
        "detail",
        "details",
        "do",
        "document",
        "documents",
        "explain",
        "fetch",
        "for",
        "from",
        "give",
        "he",
        "help",
        "her",
        "him",
        "his",
        "information",
        "inference",
        "in",
        "is",
        "it",
        "issue",
        "issues",
        "me",
        "mentioned",
        "made",
        "number",
        "of",
        "on",
        "please",
        "profile",
        "relation",
        "relations",
        "provide",
        "question",
        "questions",
        "report",
        "reports",
        "respond",
        "response",
        "rewrite",
        "relationship",
        "relationships",
        "role",
        "she",
        "show",
        "source",
        "sources",
        "summary",
        "that",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "format",
        "page",
        "pages",
        "paragraph",
        "table",
        "bullet",
        "bullets",
        "continue",
        "convert",
        "draft",
        "memo",
        "note",
        "tell",
        "the",
        "what",
        "which",
        "who",
        "with",
        "big",
        "bigdata",
        "data",
        "database",
        "db",
        "internal",
        "kb",
        "knowledge",
        "record",
        "records",
        "result",
        "results",
    }

    def __init__(self, *, known_branches: List[str], known_report_types: List[str]) -> None:
        self._known_branches = self._parse_known_values(known_branches)
        self._known_report_types = self._parse_known_values(known_report_types)
        self._branch_lookup = self._build_known_value_lookup(self._known_branches)
        self._report_type_lookup = self._build_known_value_lookup(self._known_report_types)
        self._branch_value_pattern = self._compile_known_value_pattern(self._known_branches)
        self._report_type_value_pattern = self._compile_known_value_pattern(self._known_report_types)

    @staticmethod
    def parse_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
        return None

    @staticmethod
    def safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def normalize_query_key(text: Any) -> str:
        cleaned = RetrievalSupport._SPECIAL_CHAT_TOKEN_PATTERN.sub(" ", str(text or ""))
        return " ".join(cleaned.strip().lower().split())

    @staticmethod
    def metadata_key(text: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").strip().lower()).strip()

    @staticmethod
    def clean_filter_text(value: Any) -> Optional[str]:
        text = RetrievalSupport._SPECIAL_CHAT_TOKEN_PATTERN.sub(" ", str(value or "")).strip()
        return text or None

    @classmethod
    def split_filter_values(cls, value: Any) -> List[str]:
        text = str(value or "").strip()
        if not text:
            return []
        parts = [item.strip() for item in cls._FILTER_LIST_SPLIT_PATTERN.split(text) if item.strip()]
        return parts or [text]

    @staticmethod
    def has_filter_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    @classmethod
    def _parse_known_values(cls, raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, set)):
            parts = [str(item or "").strip() for item in raw]
        else:
            parts = [item.strip() for item in cls._CSV_SPLIT_PATTERN.split(str(raw))]

        values: List[str] = []
        seen: set[str] = set()
        for item in parts:
            if not item:
                continue
            key = cls.metadata_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            values.append(item)
        return values

    @classmethod
    def _build_known_value_lookup(cls, values: List[str]) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for value in values:
            key = cls.metadata_key(value)
            if key and key not in lookup:
                lookup[key] = value
        return lookup

    @classmethod
    def _compile_known_value_pattern(cls, values: List[str]) -> Optional[re.Pattern]:
        normalized_keys: List[str] = []
        seen: set[str] = set()
        for value in values:
            key = cls.metadata_key(value)
            if not key or key in seen:
                continue
            seen.add(key)
            normalized_keys.append(key)
        if not normalized_keys:
            return None
        normalized_keys.sort(key=len, reverse=True)
        patterns: List[str] = []
        for key in normalized_keys:
            tokens = [re.escape(token) for token in key.split() if token]
            if not tokens:
                continue
            patterns.append(r"[\s_\-]*".join(tokens))
        if not patterns:
            return None
        return re.compile(rf"(?<![a-z0-9])(?:{'|'.join(patterns)})(?![a-z0-9])", flags=re.IGNORECASE)

    def match_known_value(
        self,
        text: str,
        *,
        value_pattern: Optional[re.Pattern],
        lookup: Dict[str, str],
    ) -> Optional[str]:
        if not text or not value_pattern or not lookup:
            return None
        match = value_pattern.search(text)
        if not match:
            return None
        key = self.metadata_key(match.group(0))
        return lookup.get(key)

    def extract_known_values(
        self,
        text: str,
        *,
        value_pattern: Optional[re.Pattern],
        lookup: Dict[str, str],
        max_items: int = 8,
    ) -> List[str]:
        if not text or not value_pattern or not lookup:
            return []
        out: List[str] = []
        seen: set[str] = set()
        for match in value_pattern.finditer(text):
            key = self.metadata_key(match.group(0))
            if not key or key in seen:
                continue
            canonical = lookup.get(key)
            if not canonical:
                continue
            seen.add(key)
            out.append(canonical)
            if len(out) >= max(1, int(max_items)):
                break
        return out

    def extract_labeled_known_values(
        self,
        query: str,
        *,
        field_pattern: re.Pattern,
        value_pattern: Optional[re.Pattern],
        lookup: Dict[str, str],
        max_items: int = 8,
    ) -> List[str]:
        if not query or not value_pattern or not lookup:
            return []
        out: List[str] = []
        seen: set[str] = set()
        for field_match in field_pattern.finditer(query):
            segment = str(field_match.group(1) or "").strip()
            if not segment:
                continue
            values = self.extract_known_values(
                segment,
                value_pattern=value_pattern,
                lookup=lookup,
                max_items=max_items,
            )
            segment_key = self.metadata_key(segment)
            if not values and segment_key in lookup:
                values = [lookup[segment_key]]
            for value in values:
                key = self.metadata_key(value)
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(value)
                if len(out) >= max(1, int(max_items)):
                    return out
        return out

    def canonicalize_report_type(self, value: Any) -> Optional[str]:
        text = self.clean_filter_text(value)
        if not text:
            return None
        key = self.metadata_key(text)
        if key and key in self._report_type_lookup:
            return self._report_type_lookup[key]
        matched = self.match_known_value(
            text,
            value_pattern=self._report_type_value_pattern,
            lookup=self._report_type_lookup,
        )
        return matched or text

    def normalize_report_type_values(self, raw: Any, *, max_items: int = 8) -> List[str]:
        if raw is None:
            return []
        parts: List[str] = []
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                parts.extend(self.split_filter_values(item))
        else:
            text = str(raw or "").strip()
            if not text:
                return []
            parts.extend(self.split_filter_values(text))
            if len(parts) <= 1:
                known = self.extract_known_values(
                    text,
                    value_pattern=self._report_type_value_pattern,
                    lookup=self._report_type_lookup,
                    max_items=max_items,
                )
                if known:
                    return known[: max(1, int(max_items))]
        values: List[str] = []
        seen: set[str] = set()
        for part in parts:
            canonical = self.canonicalize_report_type(part)
            if not canonical:
                continue
            key = self.metadata_key(canonical)
            if not key or key in seen:
                continue
            seen.add(key)
            values.append(canonical)
            if len(values) >= max(1, int(max_items)):
                break
        return values

    def extract_bigdata_filters_from_query(self, query: str) -> Dict[str, Any]:
        text = str(query or "").strip()
        if not text:
            return {}
        inferred: Dict[str, Any] = {}
        labeled_report_types = self.extract_labeled_known_values(
            text,
            field_pattern=self._REPORT_TYPE_FIELD_PATTERN,
            value_pattern=self._report_type_value_pattern,
            lookup=self._report_type_lookup,
            max_items=8,
        )
        fallback_report_types = self.extract_known_values(
            text,
            value_pattern=self._report_type_value_pattern,
            lookup=self._report_type_lookup,
            max_items=8,
        )
        report_types: List[str] = []
        seen_report_types: set[str] = set()
        for value in [*labeled_report_types, *fallback_report_types]:
            key = self.metadata_key(value)
            if not key or key in seen_report_types:
                continue
            seen_report_types.add(key)
            report_types.append(value)
        if report_types:
            inferred["report_types"] = report_types

        branch_values = self.extract_known_values(
            text,
            value_pattern=self._branch_value_pattern,
            lookup=self._branch_lookup,
            max_items=1,
        )
        if branch_values:
            inferred["branch"] = branch_values[0]

        doc_id_match = self._DOC_ID_FIELD_PATTERN.search(text)
        if doc_id_match:
            inferred["doc_id"] = str(doc_id_match.group(1) or "").strip()
        parent_id_match = self._PARENT_ID_FIELD_PATTERN.search(text)
        if parent_id_match:
            inferred["parent_id"] = str(parent_id_match.group(1) or "").strip()
        chunk_no_match = self._CHUNK_NO_FIELD_PATTERN.search(text)
        if chunk_no_match:
            inferred["chunk_no"] = self.safe_int(chunk_no_match.group(1))
        lang_match = self._LANG_FIELD_PATTERN.search(text)
        if lang_match:
            inferred["lang"] = str(lang_match.group(1) or "").strip().lower()
        if self._EXCLUDE_ATTACHMENT_PATTERN.search(text):
            inferred["is_attachment"] = False
        elif self._INCLUDE_ATTACHMENT_PATTERN.search(text):
            inferred["is_attachment"] = True
        return {k: v for k, v in inferred.items() if self.has_filter_value(v)}

    @classmethod
    def query_tokens(cls, query: str) -> List[str]:
        return [token for token in cls._TOKEN_PATTERN.findall(str(query or "").lower()) if token]

    @classmethod
    def query_terms(cls, query: str) -> List[str]:
        terms = [token for token in cls._TERM_PATTERN.findall(str(query or "").lower()) if len(token) >= 3]
        return list(dict.fromkeys(terms))

    @classmethod
    def specific_query_terms(cls, query: str, *, min_len: int = 4) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for token in cls.query_tokens(query):
            if len(token) < max(1, int(min_len)):
                continue
            if token.isdigit() or token in cls._LOW_SIGNAL_STOPWORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    @staticmethod
    def numeric_signature(value: Any) -> str:
        return "".join(re.findall(r"\d+", str(value or "")))

    def clean_exact_candidate(self, text: Any) -> str:
        value = self._SPECIAL_CHAT_TOKEN_PATTERN.sub(" ", str(text or "")).strip()
        if not value:
            return ""
        value = self._RETRIEVAL_SCOPE_PHRASE_PATTERN.sub(" ", value)
        value = self._EXACT_TRAILING_NOISE_PATTERN.sub("", value).strip(" \t\r\n.,:;!?-")
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _is_exact_phrase_candidate(self, text: Any) -> bool:
        value = self.clean_exact_candidate(text)
        if not value:
            return False
        if self._SOURCE_REFERENCE_PATTERN.search(value):
            return False
        numeric = self.numeric_signature(value)
        tokens = [token for token in self.query_tokens(value.lower()) if token]
        if tokens and all(token.isdigit() for token in tokens):
            return len(numeric) >= 5
        lowered = value.lower()
        if re.search(r"\bsources?\b", lowered):
            return False
        if re.search(r"\b(he|she|him|her|it|its|they|them|this|that|these|those)\b", lowered):
            return False
        if not tokens:
            return False
        signal_tokens = [token for token in tokens if token not in self._LOW_SIGNAL_STOPWORDS]
        if len(signal_tokens) == 0:
            return False
        if len(signal_tokens) > 3 and self._DESCRIPTIVE_EXACT_PHRASE_PATTERN.search(value):
            return False
        if len(numeric) >= 5:
            return True
        if len(signal_tokens) == 1:
            return len(signal_tokens[0]) >= 4
        if len(signal_tokens) == 2 and any(
            token in {"relation", "relations", "relationship", "mentioned", "about", "details", "information"}
            for token in signal_tokens
        ):
            return False
        return True

    def strip_time_filter_phrase(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        time_filter = extract_ingestion_time_filter(value)
        if not time_filter:
            return value
        matched_text = str(time_filter.matched_text or "").strip()
        if not matched_text:
            return value
        stripped = re.sub(re.escape(matched_text), " ", value, flags=re.IGNORECASE)
        stripped = re.sub(r"\s+", " ", stripped).strip(" \t\r\n,.;:-")
        return stripped or value

    def derive_entity_like_phrase(self, text: str) -> str:
        cleaned = self.strip_time_filter_phrase(self.clean_exact_candidate(text))
        if not cleaned:
            return ""
        for pattern in self._EXACT_ENTITY_CUE_PATTERNS:
            match = pattern.search(cleaned)
            if not match:
                continue
            candidate = self.clean_exact_candidate(match.group(1))
            if candidate and self._is_exact_phrase_candidate(candidate):
                return candidate
        tokens = self.specific_query_terms(cleaned, min_len=4)
        if 1 <= len(tokens) <= 4:
            candidate = " ".join(tokens)
            if not self._is_exact_phrase_candidate(candidate):
                return ""
            return " ".join(tokens)
        return ""

    def dedupe_exact_terms(self, values: List[str], *, max_items: int) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for raw in values:
            value = self.clean_exact_candidate(raw)
            if not value:
                continue
            key = self.normalize_query_key(value)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= max(1, int(max_items)):
                break
        return out

    def fallback_exact_terms(self, query: str, plan: RetrievalPlan, *, max_items: int) -> List[str]:
        candidates: List[str] = []
        cleaned_query = self.strip_time_filter_phrase(query)
        for match in self._QUOTED_PHRASE_PATTERN.finditer(cleaned_query):
            candidates.append(match.group(1))
        for match in self._LONG_NUMBER_PATTERN.finditer(cleaned_query):
            token = str(match.group(0) or "").strip()
            if len(self.numeric_signature(token)) < 5:
                continue
            candidates.append(token)
        if not candidates:
            for match in self._ID_LIKE_PATTERN.finditer(cleaned_query):
                token = str(match.group(0) or "").strip()
                if any(ch.isdigit() for ch in token):
                    candidates.append(token)
        entity_phrase = self.derive_entity_like_phrase(cleaned_query)
        if entity_phrase:
            candidates.append(entity_phrase)
        if plan.focus_subject and len(self.query_tokens(query)) <= 12:
            candidates.append(plan.focus_subject)
        if not candidates:
            tokens = self.specific_query_terms(cleaned_query, min_len=4)
            if 1 <= len(tokens) <= 3:
                candidate = " ".join(tokens)
                if self._is_exact_phrase_candidate(candidate):
                    candidates.append(candidate)
        filtered = [value for value in [*plan.exact_terms, *candidates] if self._is_exact_phrase_candidate(value)]
        return self.dedupe_exact_terms(filtered, max_items=max_items)

    def chunk_content(self, chunk: Dict[str, Any]) -> str:
        return str(
            chunk.get("content")
            or chunk.get("text")
            or chunk.get("chunk_text")
            or chunk.get("page_content")
            or ""
        ).strip()

    def chunk_similarity(self, chunk: Dict[str, Any]) -> float:
        raw = chunk.get("_sim_score", chunk.get("_final_score", chunk.get("score", 0.0)))
        try:
            return float(raw)
        except Exception:
            return 0.0

    def chunk_dedupe_key(self, chunk: Dict[str, Any]) -> str:
        doc_id = str(chunk.get("doc_id") or "").strip()
        parent_id = str(chunk.get("parent_id") or "").strip()
        file_id = str(chunk.get("file_id") or "").strip()
        chunk_id = str(chunk.get("chunk_no") or chunk.get("chunk_id") or "").strip()
        is_attachment = self.parse_bool(chunk.get("is_attachment"))
        attachment_key = "attachment" if is_attachment is True else "main" if is_attachment is False else "unknown"
        if doc_id or file_id or chunk_id:
            if not chunk_id:
                content = self.chunk_content(chunk)
                digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16] if content else "na"
                chunk_id = f"content:{digest}"
            return f"{attachment_key}|{doc_id}|{parent_id}|{file_id}|{chunk_id}"
        content = self.chunk_content(chunk)
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16] if content else "na"
        return f"content:{digest}"

    def dedupe_queries(self, queries: List[str], *, max_items: int) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for raw in queries:
            query = str(raw or "").strip()
            if not query:
                continue
            key = self.normalize_query_key(query)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(query)
            if len(out) >= max(1, int(max_items)):
                break
        return out

    def search_mode_tokens(self, search_mode: Any) -> set[str]:
        raw = str(search_mode or "").strip().lower()
        if not raw:
            return set()
        return {part for part in self._MODE_SPLIT_PATTERN.split(raw) if part}

    @staticmethod
    def extract_attachment_ids(data: Dict[str, Any]) -> List[str]:
        direct_ids = data.get("attachment_ids") or data.get("file_ids") or []
        if isinstance(direct_ids, (list, tuple, set)):
            ids: List[str] = []
            seen: set[str] = set()
            for item in direct_ids:
                attachment_id = str(item or "").strip()
                if attachment_id and attachment_id not in seen:
                    seen.add(attachment_id)
                    ids.append(attachment_id)
            if ids:
                return ids
        attachments = data.get("attachments") or []
        if not isinstance(attachments, list):
            return []
        ids = []
        seen = set()
        for item in attachments:
            if not isinstance(item, dict):
                continue
            attachment_id = str(item.get("attachment_id") or "").strip()
            if attachment_id and attachment_id not in seen:
                seen.add(attachment_id)
                ids.append(attachment_id)
        return ids

    def term_match_score(
        self,
        term: str,
        content: str,
        *,
        content_tokens: Optional[List[str]] = None,
    ) -> float:
        normalized_term = str(term or "").strip().lower()
        if not normalized_term:
            return 0.0
        body = str(content or "").lower()
        if not body:
            return 0.0
        if normalized_term in body:
            return 1.0
        tokens = content_tokens if content_tokens is not None else self.query_tokens(body)
        best_ratio = 0.0
        for token in tokens:
            if not token:
                continue
            if abs(len(token) - len(normalized_term)) > 2:
                continue
            ratio = difflib.SequenceMatcher(None, normalized_term, token).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                if best_ratio >= 0.95:
                    break
        if best_ratio >= 0.92:
            return 0.90
        if best_ratio >= 0.86:
            return 0.65
        return 0.0

    def rank_chunks_for_query(
        self,
        chunks: List[Dict[str, Any]],
        primary_query: str,
        *,
        secondary_queries: Optional[List[str]] = None,
        topic_hint: str = "",
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        primary_terms = self.query_terms(primary_query)
        secondary_terms = [self.query_terms(q) for q in list(secondary_queries or []) if str(q or "").strip()]
        topic_terms = self.specific_query_terms(topic_hint, min_len=4)
        primary_query_lower = str(primary_query or "").strip().lower()
        secondary_query_lowers = [str(q or "").strip().lower() for q in list(secondary_queries or []) if str(q or "").strip()]
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        has_topic_terms = bool(topic_terms)
        if has_topic_terms:
            semantic_weight = 0.46
            primary_weight = 0.22
            secondary_weight = 0.06
            topic_weight = 0.22
        else:
            semantic_weight = 0.62
            primary_weight = 0.26
            secondary_weight = 0.08
            topic_weight = 0.0
        for chunk in chunks:
            content = self.chunk_content(chunk).lower()
            if not content:
                continue
            content_tokens = self.query_tokens(content)
            semantic = self.chunk_similarity(chunk)
            try:
                exact_match_score = float(chunk.get("_exact_match_score", 0.0))
            except Exception:
                exact_match_score = 0.0
            exact_match_score = max(0.0, min(1.0, exact_match_score))
            primary_overlap = 0.0
            if primary_terms:
                primary_overlap = sum(1 for term in primary_terms if term in content) / max(1, len(primary_terms))
            secondary_overlap = 0.0
            for terms in secondary_terms:
                if not terms:
                    continue
                overlap = sum(1 for term in terms if term in content) / max(1, len(terms))
                if overlap > secondary_overlap:
                    secondary_overlap = overlap
            topic_overlap = 0.0
            if topic_terms:
                topic_overlap = sum(
                    self.term_match_score(term, content, content_tokens=content_tokens)
                    for term in topic_terms
                ) / max(1, len(topic_terms))
            phrase_bonus = 0.0
            if primary_query_lower and primary_query_lower in content:
                phrase_bonus += 0.12
            if any(q and len(q) >= 10 and q in content for q in secondary_query_lowers):
                phrase_bonus += 0.04
            query_match_count = max(1, int(chunk.get("_query_match_count", 1) or 1))
            query_match_bonus = min(0.12, 0.04 * max(0, query_match_count - 1))
            exact_bonus = (0.34 * exact_match_score) if exact_match_score > 0 else 0.0
            exact_keyword_count_bonus = min(0.10, 0.03 * max(0, len(list(chunk.get("_matched_keywords") or [])) - 1))
            try:
                main_score_boost = float(chunk.get("_main_score_boost", 0.0) or 0.0)
            except Exception:
                main_score_boost = 0.0
            main_score_boost = max(0.0, min(0.12, main_score_boost))
            score = (
                (semantic_weight * semantic)
                + (primary_weight * primary_overlap)
                + (secondary_weight * secondary_overlap)
                + (topic_weight * topic_overlap)
                + phrase_bonus
                + query_match_bonus
                + exact_bonus
                + exact_keyword_count_bonus
                + main_score_boost
            )
            if has_topic_terms and topic_overlap <= 0.01:
                score -= 0.06
            payload = dict(chunk)
            payload["_heuristic_score"] = score
            payload["_final_score"] = score
            ranked.append((score, payload))
        ranked.sort(key=lambda item: item[0], reverse=True)
        ordered = [chunk for _, chunk in ranked]
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in ordered:
            dedupe_key = self.chunk_dedupe_key(chunk)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped.append(chunk)
        return deduped

    def ensure_exact_match_presence(self, chunks: List[Dict[str, Any]], *, top_k: int, guaranteed: int) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        limit = max(1, int(top_k))
        if guaranteed <= 0:
            return chunks[:limit]
        result = chunks[:limit]
        exact_chunks = [chunk for chunk in chunks if float(chunk.get("_exact_match_score", 0.0) or 0.0) > 0.0]
        if not exact_chunks:
            return result

        result_keys = {self.chunk_dedupe_key(chunk) for chunk in result}
        current_exact_count = sum(
            1 for chunk in result if float(chunk.get("_exact_match_score", 0.0) or 0.0) > 0.0
        )
        needed = max(0, min(int(guaranteed), limit) - current_exact_count)
        if needed <= 0:
            return result

        replacements: List[Dict[str, Any]] = []
        for chunk in exact_chunks:
            dedupe_key = self.chunk_dedupe_key(chunk)
            if dedupe_key in result_keys:
                continue
            result_keys.add(dedupe_key)
            replacements.append(chunk)
            if len(replacements) >= needed:
                break
        if not replacements:
            return result

        for replacement in replacements:
            for index in range(len(result) - 1, -1, -1):
                if float(result[index].get("_exact_match_score", 0.0) or 0.0) <= 0.0:
                    result[index] = replacement
                    break
        return result

    def topic_alignment_score(self, chunks: List[Dict[str, Any]], topic_hint: str, *, top_k: int = 6) -> float:
        topic_terms = self.specific_query_terms(topic_hint, min_len=4)
        if not topic_terms or not chunks:
            return 0.0
        scored: List[float] = []
        for chunk in chunks[: max(1, int(top_k))]:
            content = self.chunk_content(chunk).lower()
            if not content:
                continue
            content_tokens = self.query_tokens(content)
            overlap = sum(
                self.term_match_score(term, content, content_tokens=content_tokens)
                for term in topic_terms
            ) / max(1, len(topic_terms))
            scored.append(overlap)
        if not scored:
            return 0.0
        best = max(scored)
        avg = sum(scored) / max(1, len(scored))
        return (0.65 * best) + (0.35 * avg)

    def merge_chunks_by_identity(self, query_to_chunks: List[Tuple[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for query, chunks in query_to_chunks:
            for chunk in chunks:
                payload = dict(chunk)
                key = self.chunk_dedupe_key(payload)
                if key not in merged:
                    payload["_retrieval_queries"] = [query]
                    payload["_query_match_count"] = 1
                    merged[key] = payload
                    continue
                existing = merged[key]
                existing_sim = self.chunk_similarity(existing)
                incoming_sim = self.chunk_similarity(payload)
                if incoming_sim > existing_sim:
                    existing["_sim_score"] = incoming_sim
                existing["_exact_match_score"] = max(
                    float(existing.get("_exact_match_score", 0.0) or 0.0),
                    float(payload.get("_exact_match_score", 0.0) or 0.0),
                )
                existing["_final_score"] = max(
                    float(existing.get("_final_score", 0.0) or 0.0),
                    float(payload.get("_final_score", 0.0) or 0.0),
                )
                query_list = list(existing.get("_retrieval_queries") or [])
                if query not in query_list:
                    query_list.append(query)
                existing["_retrieval_queries"] = query_list
                existing["_query_match_count"] = len(query_list)
                existing_matches = list(existing.get("_matched_keywords") or [])
                existing_match_keys = {self.normalize_query_key(value) for value in existing_matches}
                for keyword in list(payload.get("_matched_keywords") or []):
                    keyword_key = self.normalize_query_key(keyword)
                    if not keyword_key or keyword_key in existing_match_keys:
                        continue
                    existing_match_keys.add(keyword_key)
                    existing_matches.append(keyword)
                if existing_matches:
                    existing["_matched_keywords"] = existing_matches
        return list(merged.values())

    def chat_turn_count(self, chat_context: Dict[str, Any]) -> int:
        ordered_turns = list(chat_context.get("ordered_turns") or [])
        if ordered_turns:
            return len(ordered_turns)
        count = 0
        recent_pairs = list(chat_context.get("recent_conversations") or [])
        count += len([item for item in recent_pairs if isinstance(item, dict)])
        history = list(chat_context.get("conversation_history") or [])
        if count == 0 and history:
            count += max(0, len(history) // 2)
        return count

    def resolve_time_filter(self, query: str, plan: RetrievalPlan) -> Dict[str, Any]:
        if plan.time_filter:
            payload = dict(plan.time_filter)
            return payload if should_apply_retrieval_time_filter(query, payload) else {}
        parsed = extract_ingestion_time_filter(query)
        if not parsed:
            return {}
        if not should_apply_retrieval_time_filter(query, parsed):
            return {}
        return {
            "field": str(getattr(parsed, "field", "ingestion_date") or "ingestion_date"),
            "requested_field": str(getattr(parsed, "field", "ingestion_date") or "ingestion_date"),
            "label": str(getattr(parsed, "label", "") or getattr(parsed, "matched_text", "") or "").strip(),
            "matched_text": str(getattr(parsed, "matched_text", "") or "").strip(),
            "start_ms": int(getattr(parsed, "start_ms", 0) or 0),
            "end_ms": int(getattr(parsed, "end_ms", 0) or 0),
        }

    def build_no_context_message(self, *, retrieval_plan: Optional[Dict[str, Any]] = None) -> str:
        plan = dict(retrieval_plan or {})
        time_filter = dict(plan.get("time_filter") or {})
        time_label = str(time_filter.get("label") or time_filter.get("matched_text") or "").strip()
        if time_label:
            return (
                "I could not find any knowledge-base documents matching the requested filters for "
                f"`{time_label}`. Try a broader time range or fewer filters."
            )
        return (
            "I could not find relevant knowledge-base context for this request. "
            "Please refine the query or provide more specific metadata (report type, branch, doc id, date range)."
        )
