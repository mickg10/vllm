# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import (
        ResponsesRequest,
    )
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any


class KimiK2ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Kimi K2 model.

    The Kimi K2 model uses <think>...</think> tokens to denote reasoning text,
    and may implicitly end reasoning by starting a tool call section using
    <|tool_calls_section_begin|>.
    Thinking may also begin without a <think> token.

    Kimi's thinking mode can be disabled via chat_template_kwargs.
    """

    # Both plural and singular variants are recognized by the tool parser
    _TOOL_SECTION_TOKENS = (
        "<|tool_calls_section_begin|>",
        "<|tool_call_section_begin|>",
    )

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        # Token definitions
        self._start_token = "<think>"
        self._end_token = "</think>"
        self._tool_section_start_token = "<|tool_calls_section_begin|>"

        # Get token IDs
        self._start_token_id = self.vocab.get(self._start_token)
        self._end_token_id = self.vocab.get(self._end_token)
        self._tool_section_start_token_ids: set[int] = set()
        for tok in self._TOOL_SECTION_TOKENS:
            tid = self.vocab.get(tok)
            if tid is not None:
                self._tool_section_start_token_ids.add(tid)

        if self._start_token_id is None or self._end_token_id is None:
            raise RuntimeError(
                "KimiK2ReasoningParser could not locate think start/end "
                "tokens in the tokenizer!"
            )

        # Check if thinking is disabled via chat_template_kwargs
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = bool(chat_kwargs.get("thinking", True))

        # If thinking is not enabled, use identity parser to fall through
        self._identity_parser: IdentityReasoningParser | None
        if not thinking:
            self._identity_parser = IdentityReasoningParser(
                tokenizer, *args, **kwargs
            )
        else:
            self._identity_parser = None

    def _is_identity_mode(self) -> bool:
        """Check if parser is in identity mode (no reasoning extraction)."""
        return self._identity_parser is not None

    def _has_tool_section_token(self, token_ids: Sequence[int]) -> bool:
        """Check if any tool section start token is in the token_ids."""
        return bool(self._tool_section_start_token_ids.intersection(token_ids))

    def _is_tool_section_token(self, token_id: int) -> bool:
        """Check if a single token_id is a tool section start token."""
        return token_id in self._tool_section_start_token_ids

    def _find_earliest_end(self, text: str) -> tuple[int, int]:
        """Find the earliest reasoning-end marker in text.

        Returns (index, skip_length) where:
        - index is the position of the marker (-1 if not found)
        - skip_length is how many chars to skip over:
          - For </think>: skip the full tag
          - For tool section markers: skip 0 (tool parser needs them)
        """
        think_idx = text.find(self._end_token)

        tool_idx = -1
        for tok in self._TOOL_SECTION_TOKENS:
            idx = text.find(tok)
            if idx != -1 and (tool_idx == -1 or idx < tool_idx):
                tool_idx = idx

        if think_idx == -1 and tool_idx == -1:
            return (-1, 0)
        if think_idx == -1:
            return (tool_idx, 0)
        if tool_idx == -1:
            return (think_idx, len(self._end_token))
        # Both found — use whichever comes first
        if think_idx <= tool_idx:
            return (think_idx, len(self._end_token))
        return (tool_idx, 0)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """
        Check if the most recent reasoning block has ended.

        Reasoning ends when we see either:
        1. The end token (</think>)
        2. A tool section start token (<|tool_calls_section_begin|>)
        """
        if self._is_identity_mode():
            assert self._identity_parser is not None
            return self._identity_parser.is_reasoning_end(input_ids)

        start_token_id = self._start_token_id
        end_token_id = self._end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == end_token_id:
                return True
            if self._is_tool_section_token(input_ids[i]):
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        """
        Check if the reasoning content ends in the delta on a decode step.
        """
        if self._is_identity_mode():
            assert self._identity_parser is not None
            return self._identity_parser.is_reasoning_end_streaming(
                input_ids, delta_ids
            )

        if self._end_token_id in delta_ids:
            return True
        return self._has_tool_section_token(delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        if self._is_identity_mode():
            assert self._identity_parser is not None
            return self._identity_parser.extract_content_ids(input_ids)

        if self._end_token_id in input_ids:
            end_token_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._end_token_id)
            )
            return input_ids[end_token_index + 1 :]

        # Check tool section tokens (include the marker — tool parser needs it)
        for tid in self._tool_section_start_token_ids:
            if tid in input_ids:
                tool_section_index = (
                    len(input_ids) - 1 - input_ids[::-1].index(tid)
                )
                return input_ids[tool_section_index:]

        # still reasoning (no content)
        return []

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
        """
        if self._is_identity_mode():
            assert self._identity_parser is not None
            return self._identity_parser.extract_reasoning(
                model_output, request  # type: ignore[arg-type]
            )

        # Consume <think> if present, using partition for correctness
        parts = model_output.partition(self._start_token)
        text_after_start = parts[2] if parts[1] else parts[0]

        # Find earliest end marker (</think> or tool section begin)
        end_idx, skip_len = self._find_earliest_end(text_after_start)

        if end_idx != -1:
            reasoning = text_after_start[:end_idx]
            content = text_after_start[end_idx + skip_len:]
            return (reasoning, content or None)

        # No end marker — still reasoning
        return (text_after_start, None)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message during streaming.
        """
        if self._is_identity_mode():
            assert self._identity_parser is not None
            return self._identity_parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        # If reasoning has already ended in previous tokens, this is content
        if self.is_reasoning_end(previous_token_ids):
            return DeltaMessage(content=delta_text)

        # Skip single special tokens (start, end, tool section)
        if len(delta_token_ids) == 1:
            tid = delta_token_ids[0]
            if tid in (self._start_token_id, self._end_token_id):
                return None
            if self._is_tool_section_token(tid):
                # Tool section marker as sole delta — pass as content
                return DeltaMessage(content=delta_text)

        # Handle <think> in multi-token delta: strip it from reasoning
        if self._start_token_id in delta_token_ids:
            if self._end_token_id in delta_token_ids:
                # Both start and end in same delta
                start_index = delta_text.find(self._start_token)
                end_index = delta_text.find(self._end_token)
                reasoning = delta_text[
                    start_index + len(self._start_token) : end_index
                ]
                content = delta_text[end_index + len(self._end_token) :]
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content if content else None,
                )
            if self._has_tool_section_token(delta_token_ids):
                # Start token and tool section in same delta
                end_idx, skip_len = self._find_earliest_end(delta_text)
                if end_idx != -1:
                    start_index = delta_text.find(self._start_token)
                    reasoning = delta_text[
                        start_index + len(self._start_token) : end_idx
                    ]
                    content = delta_text[end_idx + skip_len:]
                    return DeltaMessage(
                        reasoning=reasoning or None,
                        content=content if content else None,
                    )
            # Start token with ongoing reasoning
            return DeltaMessage(reasoning=delta_text)

        # Handle </think> in delta
        if self._end_token_id in delta_token_ids:
            end_index = delta_text.find(self._end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self._end_token) :]
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content if content else None,
            )

        # Handle tool section start in delta
        if self._has_tool_section_token(delta_token_ids):
            end_idx, _ = self._find_earliest_end(delta_text)
            if end_idx != -1:
                reasoning = delta_text[:end_idx]
                content = delta_text[end_idx:]
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content if content else None,
                )

        # still reasoning (no end token)
        return DeltaMessage(reasoning=delta_text)
