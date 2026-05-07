import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from context_creator import ContextFetcher

logger = logging.getLogger(__name__)


@dataclass
class Config:
    template_path: str = "jinja_template.txt"
    max_recent_conversations: int = 5
    max_history_conversations: Optional[int] = 10


class ChatContextNewFetcher:
    def __init__(self, context_fetcher: ContextFetcher, cfg: Optional[Config] = None):
        self.context_fetcher = context_fetcher
        self.cfg = cfg or Config()

    async def fetch(
        self,
        *,
        user_id: str,
        chat_id: str,
        message_id: str,
        query: str,
        top_n: int = 9,
        semantic_threshold: Optional[float] = None,
        enable_semantic_search: bool = True,
        tail_max_turns: Optional[int] = None,
        tail_token_budget: Optional[int] = None,
        semantic_history_max_turns: Optional[int] = None,
        semantic_history_token_budget: Optional[int] = None,
        total_token_budget: Optional[int] = None,
        chars_per_token: Optional[float] = None,
        include_chronological_anchors: Optional[bool] = None,
        chronological_anchor_turns: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not user_id or not chat_id or not message_id:
            raise ValueError("user_id, chat_id and message_id are required")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        normalized_query = str(query or "").strip()
        effective_enable_semantic_search = bool(enable_semantic_search and normalized_query)

        logger.info(
            "chat_context_new fetch request user_id=%s chat_id=%s message_id=%s query_len=%s top_n=%s semantic_threshold=%s enable_semantic_search=%s",
            user_id,
            chat_id,
            message_id,
            len(normalized_query),
            top_n,
            semantic_threshold,
            effective_enable_semantic_search,
        )

        messages = await self.context_fetcher.fetch(
            user_id=user_id,
            chat_id=chat_id,
            message_id=message_id,
            query=normalized_query,
            top_n=top_n,
            semantic_threshold=semantic_threshold,
            enable_semantic_search=effective_enable_semantic_search,
            tail_max_turns=tail_max_turns,
            tail_token_budget=tail_token_budget,
            semantic_history_max_turns=semantic_history_max_turns,
            semantic_history_token_budget=semantic_history_token_budget,
            total_token_budget=total_token_budget,
            chars_per_token=chars_per_token,
            include_chronological_anchors=include_chronological_anchors,
            chronological_anchor_turns=chronological_anchor_turns,
        )
        logger.info(
            "chat_context_new base context fetched chat_id=%s message_count=%s",
            chat_id,
            len(messages),
        )
        all_pairs = self._build_conversation_pairs(messages)
        if self._has_scope_tags(all_pairs):
            history_pairs, recent_conversations = self._split_pairs_by_scope(all_pairs)
            conversation_history = self._history_messages_from_pairs(history_pairs)
        else:
            conversation_history, recent_conversations = self._split_history_and_recent(
                pairs=all_pairs,
                max_recent=self.cfg.max_recent_conversations,
                max_history=self.cfg.max_history_conversations,
            )
            history_pairs = self._history_pairs_from_messages(conversation_history)
        ordered_turns = self._build_ordered_turns(
            history_pairs=history_pairs,
            recent_pairs=recent_conversations,
        )
        logger.info(
            "chat_context_new split pairs chat_id=%s total_pairs=%s history_pairs=%s recent_pairs=%s ordered_turns=%s",
            chat_id,
            len(all_pairs),
            len(conversation_history) // 2,
            len(recent_conversations),
            len(ordered_turns),
        )

        return {
            "conversation_history": conversation_history,
            "recent_conversations": recent_conversations,
            "ordered_turns": ordered_turns,
            "user_query": query,
            "message_count": len(messages),
            "recent_conversations_count": len(recent_conversations),
            "conversation_history_count": len(conversation_history),
            "ordered_turns_count": len(ordered_turns),
        }

    @staticmethod
    def _normalize_attachment_ids(value: Any) -> List[str]:
        if value is None:
            return []
        items: List[Any]
        if isinstance(value, (list, tuple, set)):
            items = list(value)
        elif isinstance(value, str):
            text = str(value).strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    items = parsed
                else:
                    items = [part.strip() for part in text.split(",")]
            else:
                items = [part.strip() for part in text.split(",")]
        else:
            items = [value]

        out: List[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = str(item or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out

    def _build_conversation_pairs(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(messages):
            current = messages[idx]
            role = str(current.get("role", "")).strip().lower()
            content = str(current.get("content", "")).strip()
            current_attachment_ids = self._normalize_attachment_ids(current.get("attachment_ids"))

            if role == "user" and (content or current_attachment_ids) and idx + 1 < len(messages):
                nxt = messages[idx + 1]
                next_role = str(nxt.get("role", "")).strip().lower()
                next_content = str(nxt.get("content", "")).strip()
                if next_role in {"system", "assistant"}:
                    user_attachment_ids = current_attachment_ids
                    assistant_attachment_ids = self._normalize_attachment_ids(nxt.get("attachment_ids"))
                    pairs.append(
                        {
                            "turn_id": current.get("_turn_id") or nxt.get("_turn_id") or len(pairs),
                            "scope": str(nxt.get("_turn_scope") or current.get("_turn_scope") or current.get("_scope") or ""),
                            "is_recent": bool(nxt.get("_turn_is_recent") or current.get("_turn_is_recent")),
                            "created_at": nxt.get("created_at") or current.get("created_at"),
                            "user_message_id": current.get("message_id"),
                            "assistant_message_id": nxt.get("message_id"),
                            "user_seq": current.get("seq"),
                            "assistant_seq": nxt.get("seq"),
                            "user_message": content,
                            "assistant_reply": next_content,
                            "attachment_ids": list(user_attachment_ids),
                            "has_attachments": bool(user_attachment_ids or assistant_attachment_ids),
                            "user_attachment_ids": list(user_attachment_ids),
                            "assistant_attachment_ids": list(assistant_attachment_ids),
                        }
                    )
                    idx += 2
                    continue
            if role == "user" and (content or current_attachment_ids):
                pairs.append(
                    {
                        "turn_id": current.get("_turn_id") or len(pairs),
                        "scope": str(current.get("_turn_scope") or current.get("_scope") or ""),
                        "is_recent": bool(current.get("_turn_is_recent")),
                        "created_at": current.get("created_at"),
                        "user_message_id": current.get("message_id"),
                        "assistant_message_id": "",
                        "user_seq": current.get("seq"),
                        "assistant_seq": None,
                        "user_message": content,
                        "assistant_reply": "",
                        "attachment_ids": list(current_attachment_ids),
                        "has_attachments": bool(current_attachment_ids),
                        "user_attachment_ids": list(current_attachment_ids),
                        "assistant_attachment_ids": [],
                    }
                )
                idx += 1
                continue
            idx += 1

        return pairs

    @staticmethod
    def _has_scope_tags(pairs: List[Dict[str, Any]]) -> bool:
        for pair in pairs:
            scope = str(pair.get("scope") or "").strip()
            if scope or bool(pair.get("is_recent")):
                return True
        return False

    @staticmethod
    def _is_recent_pair(pair: Dict[str, Any]) -> bool:
        scope = str(pair.get("scope") or "").strip().lower()
        if scope in {"recent", "tail", "recent_cache", "tail_recent"}:
            return True
        if scope in {"semantic", "history", "semantic_history"}:
            return False
        return bool(pair.get("is_recent"))

    def _split_pairs_by_scope(
        self,
        pairs: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        history_pairs: List[Dict[str, Any]] = []
        recent_pairs: List[Dict[str, Any]] = []
        for pair in pairs:
            payload = dict(pair)
            if self._is_recent_pair(payload):
                payload["is_recent"] = True
                if not str(payload.get("scope") or "").strip():
                    payload["scope"] = "recent"
                recent_pairs.append(payload)
            else:
                payload["is_recent"] = False
                if not str(payload.get("scope") or "").strip():
                    payload["scope"] = "semantic"
                history_pairs.append(payload)
        return history_pairs, recent_pairs

    @staticmethod
    def _history_messages_from_pairs(
        pairs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        history_messages: List[Dict[str, Any]] = []
        for pair in pairs:
            history_messages.append(
                {
                    "role": "user",
                    "text": pair.get("user_message", ""),
                    "message_id": pair.get("user_message_id"),
                    "created_at": pair.get("created_at"),
                    "seq": pair.get("user_seq"),
                    "attachment_ids": list(pair.get("user_attachment_ids") or pair.get("attachment_ids") or []),
                    "has_attachments": bool(pair.get("user_attachment_ids") or pair.get("attachment_ids")),
                }
            )
            history_messages.append(
                {
                    "role": "assistant",
                    "text": pair.get("assistant_reply", ""),
                    "message_id": pair.get("assistant_message_id"),
                    "created_at": pair.get("created_at"),
                    "seq": pair.get("assistant_seq"),
                    "attachment_ids": list(pair.get("assistant_attachment_ids") or []),
                    "has_attachments": bool(pair.get("assistant_attachment_ids")),
                }
            )
        return history_messages

    def _split_history_and_recent(
        self,
        *,
        pairs: List[Dict[str, Any]],
        max_recent: int,
        max_history: Optional[int],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if max_recent < 0:
            raise ValueError("max_recent_conversations must be >= 0")
        if max_history is not None and max_history < 0:
            raise ValueError("max_history_conversations must be >= 0")

        if max_recent == 0:
            recent_pairs: List[Dict[str, str]] = []
            older_pairs = list(pairs)
        elif len(pairs) <= max_recent:
            recent_pairs = list(pairs)
            older_pairs = []
        else:
            recent_pairs = pairs[-max_recent:]
            older_pairs = pairs[:-max_recent]

        if max_history is not None:
            older_pairs = older_pairs[-max_history:]

        history_messages: List[Dict[str, Any]] = []
        for pair in older_pairs:
            history_messages.append(
                {
                    "role": "user",
                    "text": pair.get("user_message", ""),
                    "message_id": pair.get("user_message_id"),
                    "created_at": pair.get("created_at"),
                    "seq": pair.get("user_seq"),
                    "attachment_ids": list(pair.get("user_attachment_ids") or pair.get("attachment_ids") or []),
                    "has_attachments": bool(pair.get("user_attachment_ids") or pair.get("attachment_ids")),
                }
            )
            history_messages.append(
                {
                    "role": "assistant",
                    "text": pair.get("assistant_reply", ""),
                    "message_id": pair.get("assistant_message_id"),
                    "created_at": pair.get("created_at"),
                    "seq": pair.get("assistant_seq"),
                    "attachment_ids": list(pair.get("assistant_attachment_ids") or []),
                    "has_attachments": bool(pair.get("assistant_attachment_ids")),
                }
            )

        return history_messages, recent_pairs

    def _history_pairs_from_messages(
        self,
        history_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(history_messages):
            current = history_messages[idx]
            role = str(current.get("role", "")).strip().lower()
            text = str(current.get("text", "")).strip()
            current_attachment_ids = self._normalize_attachment_ids(current.get("attachment_ids"))
            if role == "user" and (text or current_attachment_ids) and idx + 1 < len(history_messages):
                nxt = history_messages[idx + 1]
                next_role = str(nxt.get("role", "")).strip().lower()
                next_text = str(nxt.get("text", "")).strip()
                if next_role in {"assistant", "system"}:
                    pairs.append(
                        {
                            "turn_id": len(pairs),
                            "scope": "semantic",
                            "is_recent": False,
                            "created_at": nxt.get("created_at") or current.get("created_at"),
                            "user_message_id": current.get("message_id"),
                            "assistant_message_id": nxt.get("message_id"),
                            "user_seq": current.get("seq"),
                            "assistant_seq": nxt.get("seq"),
                            "user_message": text,
                            "assistant_reply": next_text,
                            "attachment_ids": current_attachment_ids,
                            "has_attachments": bool(current.get("attachment_ids") or nxt.get("attachment_ids")),
                            "user_attachment_ids": current_attachment_ids,
                            "assistant_attachment_ids": self._normalize_attachment_ids(nxt.get("attachment_ids")),
                        }
                    )
                    idx += 2
                    continue
            if role == "user" and (text or current_attachment_ids):
                pairs.append(
                    {
                        "turn_id": len(pairs),
                        "scope": "semantic",
                        "is_recent": False,
                        "created_at": current.get("created_at"),
                        "user_message_id": current.get("message_id"),
                        "assistant_message_id": "",
                        "user_seq": current.get("seq"),
                        "assistant_seq": None,
                        "user_message": text,
                        "assistant_reply": "",
                        "attachment_ids": current_attachment_ids,
                        "has_attachments": bool(current_attachment_ids),
                        "user_attachment_ids": current_attachment_ids,
                        "assistant_attachment_ids": [],
                    }
                )
                idx += 1
                continue
            idx += 1
        return pairs

    def _build_ordered_turns(
        self,
        *,
        history_pairs: List[Dict[str, Any]],
        recent_pairs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        combined = list(history_pairs) + list(recent_pairs)
        total = len(combined)
        turns: List[Dict[str, Any]] = []
        for idx, pair in enumerate(combined, start=1):
            turns.append(
                {
                    "turn_index": idx,
                    "recency_rank": (total - idx) + 1,
                    "is_recent": bool(pair.get("is_recent")),
                    "scope": str(pair.get("scope") or ("recent" if pair.get("is_recent") else "history")),
                    "created_at": pair.get("created_at"),
                    "user_message_id": pair.get("user_message_id"),
                    "assistant_message_id": pair.get("assistant_message_id"),
                    "user_seq": pair.get("user_seq"),
                    "assistant_seq": pair.get("assistant_seq"),
                    "user_message": str(pair.get("user_message") or "").strip(),
                    "assistant_reply": str(pair.get("assistant_reply") or "").strip(),
                    "attachment_ids": list(pair.get("attachment_ids") or pair.get("user_attachment_ids") or []),
                    "has_attachments": bool(pair.get("has_attachments")),
                    "user_attachment_ids": list(pair.get("user_attachment_ids") or pair.get("attachment_ids") or []),
                    "assistant_attachment_ids": list(pair.get("assistant_attachment_ids") or []),
                }
            )
        return turns

    def _render_context(
        self,
        *,
        conversation_history: List[Dict[str, str]],
        recent_conversations: List[Dict[str, str]],
        user_query: str,
    ) -> str:
        template_name = Path(self.cfg.template_path).name
        logger.info("Rendering chat context with template=%s", template_name)

        lines: List[str] = []
        lines.append("### Conversation History:")
        for message in conversation_history:
            role = message.get("role", "assistant").capitalize()
            text = message.get("text", "")
            lines.append(f"- **{role}**: {text}")

        lines.append("")
        lines.append("### Recent Conversations:")
        for convo in recent_conversations:
            lines.append(f"- **User**: {convo.get('user_message', '')}")
            lines.append(f"- **Assistant**: {convo.get('assistant_reply', '')}")

        lines.append("")
        lines.append("### Current Query:")
        lines.append(f"User's current query: \"{user_query}\"")
        return "\n".join(lines)
