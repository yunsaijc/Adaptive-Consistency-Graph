from __future__ import annotations

import json


class TokenCounter:
    def __init__(self, tokenizer_path: str | None = None):
        self.tokenizer = None
        if tokenizer_path:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True
            )

    def count_text(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.encode("utf-8"))
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count(self, messages: list[dict], tools: list[dict]) -> int:
        if self.tokenizer is not None:
            try:
                return len(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tools=tools or None,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
            except (ValueError, TypeError, AttributeError):
                pass
        payload = json.dumps(
            {"messages": messages, "tools": tools or [], "add_generation_prompt": True},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return self.count_text(payload)
