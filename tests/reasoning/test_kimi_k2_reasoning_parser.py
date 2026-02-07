# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for KimiK2ReasoningParser.

Since the Kimi K2 tokenizer (moonshotai/Kimi-K2-Instruct) is very large,
these tests use a mock tokenizer with the required special tokens.
"""

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.kimi_k2_reasoning_parser import KimiK2ReasoningParser

# Token IDs for mock tokenizer
THINK_START_ID = 100
THINK_END_ID = 101
TOOL_SECTION_BEGIN_ID = 102
TOOL_SECTION_BEGIN_SINGULAR_ID = 103
TOOL_CALL_BEGIN_ID = 104
TOOL_CALL_ARG_BEGIN_ID = 105
TOOL_CALL_END_ID = 106
TOOL_SECTION_END_ID = 107
# Regular token IDs
TOKEN_A_ID = 200
TOKEN_B_ID = 201
TOKEN_C_ID = 202

MOCK_VOCAB = {
    "<think>": THINK_START_ID,
    "</think>": THINK_END_ID,
    "<|tool_calls_section_begin|>": TOOL_SECTION_BEGIN_ID,
    "<|tool_call_section_begin|>": TOOL_SECTION_BEGIN_SINGULAR_ID,
    "<|tool_call_begin|>": TOOL_CALL_BEGIN_ID,
    "<|tool_call_argument_begin|>": TOOL_CALL_ARG_BEGIN_ID,
    "<|tool_call_end|>": TOOL_CALL_END_ID,
    "<|tool_calls_section_end|>": TOOL_SECTION_END_ID,
    "Hello": TOKEN_A_ID,
    "World": TOKEN_B_ID,
    "Reason": TOKEN_C_ID,
}


def _make_mock_tokenizer():
    """Create a mock tokenizer with Kimi K2 special tokens."""
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = MOCK_VOCAB
    # tokenize returns a list of token strings
    tokenizer.tokenize.side_effect = lambda text: [
        tok for tok in MOCK_VOCAB if tok in text
    ]
    return tokenizer


@pytest.fixture(scope="module")
def parser():
    """Create a KimiK2ReasoningParser with mock tokenizer."""
    tokenizer = _make_mock_tokenizer()
    return KimiK2ReasoningParser(tokenizer)


@pytest.fixture
def chat_request():
    return ChatCompletionRequest(messages=[], model="test-model")


# --- extract_reasoning (non-streaming) ---


class TestExtractReasoning:
    def test_think_tags(self, parser, chat_request):
        """Standard <think>...</think> parsing."""
        reasoning, content = parser.extract_reasoning(
            "<think>I need to think</think>The answer is 42.", chat_request
        )
        assert reasoning == "I need to think"
        assert content == "The answer is 42."

    def test_no_think_start(self, parser, chat_request):
        """Reasoning without <think> start tag."""
        reasoning, content = parser.extract_reasoning(
            "I need to think</think>The answer is 42.", chat_request
        )
        assert reasoning == "I need to think"
        assert content == "The answer is 42."

    def test_tool_call_ends_reasoning(self, parser, chat_request):
        """<|tool_calls_section_begin|> ends reasoning without </think>."""
        text = (
            "<think>I need to call a function"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0"
            '<|tool_call_argument_begin|>{"city":"NYC"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        reasoning, content = parser.extract_reasoning(text, chat_request)
        assert reasoning == "I need to call a function"
        assert content is not None
        assert "<|tool_calls_section_begin|>" in content

    def test_think_end_before_tool_call(self, parser, chat_request):
        """</think> before tool call — </think> takes priority."""
        text = (
            "<think>reasoning</think>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.foo:0"
        )
        reasoning, content = parser.extract_reasoning(text, chat_request)
        assert reasoning == "reasoning"
        assert content is not None
        assert "<|tool_calls_section_begin|>" in content

    def test_tool_call_before_think_end(self, parser, chat_request):
        """Tool call marker before </think> — tool marker takes priority."""
        text = (
            "<think>reasoning"
            "<|tool_calls_section_begin|>tool_stuff"
            "</think>more"
        )
        reasoning, content = parser.extract_reasoning(text, chat_request)
        assert reasoning == "reasoning"
        assert content is not None
        assert content.startswith("<|tool_calls_section_begin|>")

    def test_no_end_marker(self, parser, chat_request):
        """No end marker — all reasoning, no content."""
        reasoning, content = parser.extract_reasoning(
            "<think>still thinking", chat_request
        )
        assert reasoning == "still thinking"
        assert content is None

    def test_empty_reasoning(self, parser, chat_request):
        """Empty reasoning block."""
        reasoning, content = parser.extract_reasoning(
            "<think></think>answer", chat_request
        )
        assert reasoning == ""
        assert content == "answer"

    def test_think_at_nonzero_position(self, parser, chat_request):
        """<think> not at position 0 — should still extract correctly."""
        reasoning, content = parser.extract_reasoning(
            "\n<think>reasoning</think>answer", chat_request
        )
        assert reasoning == "reasoning"
        assert content == "answer"


# --- is_reasoning_end ---


class TestIsReasoningEnd:
    def test_end_token(self, parser):
        ids = [THINK_START_ID, TOKEN_A_ID, THINK_END_ID, TOKEN_B_ID]
        assert parser.is_reasoning_end(ids) is True

    def test_tool_section_token(self, parser):
        ids = [THINK_START_ID, TOKEN_A_ID, TOOL_SECTION_BEGIN_ID, TOKEN_B_ID]
        assert parser.is_reasoning_end(ids) is True

    def test_singular_tool_section_token(self, parser):
        ids = [THINK_START_ID, TOKEN_A_ID, TOOL_SECTION_BEGIN_SINGULAR_ID]
        assert parser.is_reasoning_end(ids) is True

    def test_still_reasoning(self, parser):
        ids = [THINK_START_ID, TOKEN_A_ID, TOKEN_B_ID]
        assert parser.is_reasoning_end(ids) is False

    def test_reopened_reasoning(self, parser):
        """Second <think> after </think> — latest block is open."""
        ids = [
            THINK_START_ID, TOKEN_A_ID, THINK_END_ID,
            TOKEN_B_ID, THINK_START_ID, TOKEN_C_ID,
        ]
        assert parser.is_reasoning_end(ids) is False

    def test_empty(self, parser):
        assert parser.is_reasoning_end([]) is False


# --- is_reasoning_end_streaming ---


class TestIsReasoningEndStreaming:
    def test_end_token_in_delta(self, parser):
        assert parser.is_reasoning_end_streaming([], [THINK_END_ID]) is True

    def test_tool_section_in_delta(self, parser):
        assert parser.is_reasoning_end_streaming(
            [], [TOOL_SECTION_BEGIN_ID]
        ) is True

    def test_singular_tool_section_in_delta(self, parser):
        assert parser.is_reasoning_end_streaming(
            [], [TOOL_SECTION_BEGIN_SINGULAR_ID]
        ) is True

    def test_normal_token(self, parser):
        assert parser.is_reasoning_end_streaming([], [TOKEN_A_ID]) is False


# --- extract_content_ids ---


class TestExtractContentIds:
    def test_after_end_token(self, parser):
        ids = [THINK_START_ID, TOKEN_A_ID, THINK_END_ID, TOKEN_B_ID]
        assert parser.extract_content_ids(ids) == [TOKEN_B_ID]

    def test_tool_section_included(self, parser):
        """Tool section marker is included in content (tool parser needs it)."""
        ids = [THINK_START_ID, TOKEN_A_ID, TOOL_SECTION_BEGIN_ID, TOKEN_B_ID]
        assert parser.extract_content_ids(ids) == [
            TOOL_SECTION_BEGIN_ID, TOKEN_B_ID
        ]

    def test_end_token_preferred(self, parser):
        """</think> takes precedence over tool section."""
        ids = [
            THINK_START_ID, TOKEN_A_ID, THINK_END_ID,
            TOOL_SECTION_BEGIN_ID, TOKEN_B_ID,
        ]
        assert parser.extract_content_ids(ids) == [
            TOOL_SECTION_BEGIN_ID, TOKEN_B_ID
        ]

    def test_still_reasoning(self, parser):
        ids = [THINK_START_ID, TOKEN_A_ID, TOKEN_B_ID]
        assert parser.extract_content_ids(ids) == []


# --- extract_reasoning_streaming ---


class TestExtractReasoningStreaming:
    def _call(
        self,
        parser,
        delta_text: str,
        delta_ids: list[int],
        prev_text: str = "",
        prev_ids: list[int] | None = None,
    ) -> DeltaMessage | None:
        prev_ids = prev_ids or []
        cur_text = prev_text + delta_text
        cur_ids = prev_ids + delta_ids
        return parser.extract_reasoning_streaming(
            prev_text, cur_text, delta_text, prev_ids, cur_ids, delta_ids
        )

    def test_reasoning_already_ended(self, parser):
        """After reasoning ends, all deltas are content."""
        prev_ids = [THINK_START_ID, TOKEN_A_ID, THINK_END_ID]
        result = self._call(
            parser, "answer", [TOKEN_B_ID], prev_ids=prev_ids
        )
        assert result is not None
        assert result.content == "answer"
        assert result.reasoning is None

    def test_skip_start_token(self, parser):
        """Single <think> token is skipped."""
        result = self._call(parser, "<think>", [THINK_START_ID])
        assert result is None

    def test_skip_end_token(self, parser):
        """Single </think> token is skipped."""
        result = self._call(
            parser, "</think>", [THINK_END_ID],
            prev_ids=[THINK_START_ID, TOKEN_A_ID],
        )
        assert result is None

    def test_single_tool_section_token(self, parser):
        """Single tool section token becomes content."""
        result = self._call(
            parser,
            "<|tool_calls_section_begin|>",
            [TOOL_SECTION_BEGIN_ID],
            prev_ids=[THINK_START_ID, TOKEN_A_ID],
        )
        assert result is not None
        assert result.content == "<|tool_calls_section_begin|>"

    def test_end_token_with_content(self, parser):
        """</think> with trailing content in same delta."""
        result = self._call(
            parser, "</think>The answer", [THINK_END_ID, TOKEN_B_ID],
            prev_ids=[THINK_START_ID, TOKEN_A_ID],
        )
        assert result is not None
        assert result.content == "The answer"

    def test_tool_section_in_delta(self, parser):
        """Tool section marker in multi-token delta."""
        delta_text = "text<|tool_calls_section_begin|>tool_stuff"
        delta_ids = [TOKEN_A_ID, TOOL_SECTION_BEGIN_ID, TOKEN_B_ID]
        result = self._call(
            parser, delta_text, delta_ids,
            prev_ids=[THINK_START_ID],
        )
        assert result is not None
        assert result.reasoning is None or "text" in (result.reasoning or "")
        assert result.content is not None
        assert "<|tool_calls_section_begin|>" in result.content

    def test_ongoing_reasoning(self, parser):
        """Normal reasoning delta without end markers."""
        result = self._call(
            parser, "still thinking", [TOKEN_A_ID, TOKEN_B_ID],
            prev_ids=[THINK_START_ID],
        )
        assert result is not None
        assert result.reasoning == "still thinking"
        assert result.content is None

    def test_start_and_end_in_same_delta(self, parser):
        """<think> and </think> in same multi-token delta."""
        delta_text = "<think>reasoning</think>content"
        delta_ids = [THINK_START_ID, TOKEN_A_ID, THINK_END_ID, TOKEN_B_ID]
        result = self._call(parser, delta_text, delta_ids)
        assert result is not None
        assert result.reasoning == "reasoning"
        assert result.content == "content"


# --- identity mode ---


class TestIdentityMode:
    @pytest.fixture
    def identity_parser(self):
        tokenizer = _make_mock_tokenizer()
        return KimiK2ReasoningParser(
            tokenizer, chat_template_kwargs={"thinking": False}
        )

    def test_extract_reasoning_identity(self, identity_parser, chat_request):
        """With thinking=False, all output is content."""
        reasoning, content = identity_parser.extract_reasoning(
            "<think>reasoning</think>content", chat_request
        )
        # Identity parser returns all as content
        assert reasoning is None

    def test_is_reasoning_end_identity(self, identity_parser):
        assert identity_parser.is_reasoning_end([TOKEN_A_ID]) is True
