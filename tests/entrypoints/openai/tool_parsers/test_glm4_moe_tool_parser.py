# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest

from tests.entrypoints.openai.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager


def make_tool_call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(type="function",
                    id="call_0",
                    function=FunctionCall(name=name,
                                          arguments=json.dumps(
                                              arguments,
                                              ensure_ascii=False)))


def _make_mock_tokenizer() -> MagicMock:
    """Minimal tokenizer mock to satisfy ToolParser's vocab/token checks."""
    tok = MagicMock()
    tok.get_vocab.return_value = {
        "<tool_call>": 1,
        "</tool_call>": 2,
    }

    def tokenize(text: str):
        # Ensure start/end tags appear as their own "tokens" when present.
        parts = []
        i = 0
        while i < len(text):
            if text.startswith("<tool_call>", i):
                parts.append("<tool_call>")
                i += len("<tool_call>")
            elif text.startswith("</tool_call>", i):
                parts.append("</tool_call>")
                i += len("</tool_call>")
            else:
                # Consume until next tag boundary.
                next_start = text.find("<tool_call>", i)
                next_end = text.find("</tool_call>", i)
                next_idx = min(
                    [x for x in [next_start, next_end] if x != -1] or
                    [len(text)])
                parts.append(text[i:next_idx])
                i = next_idx
        return parts

    tok.tokenize.side_effect = tokenize
    return tok


@pytest.mark.parametrize(
    "model_output, expected_tool_calls, expected_content",
    [
        ("How can I help you today?", [], "How can I help you today?"),
        (
            "<tool_call>\n"
            "bash\n"
            "<arg_key>command</arg_key><arg_value>echo TOOL_OK</arg_value>\n"
            "</tool_call>",
            [make_tool_call("bash", {"command": "echo TOOL_OK"})],
            None,
        ),
        (
            "I will call the tool now.\n"
            "<tool_call>\n"
            "bash\n"
            "<arg_key>command</arg_key><arg_value>echo TOOL_OK</arg_value>\n"
            "</tool_call>\n"
            "Thanks!",
            [make_tool_call("bash", {"command": "echo TOOL_OK"})],
            "I will call the tool now.",
        ),
    ],
)
def test_glm4_moe_tool_parser_extract(model_output, expected_tool_calls,
                                      expected_content):
    mock_tokenizer = _make_mock_tokenizer()
    tool_parser: ToolParser = ToolParserManager.get_tool_parser(
        "glm47")(mock_tokenizer)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=False)

    assert tool_calls == expected_tool_calls
    assert content == expected_content


def test_glm4_moe_tool_parser_streaming_single_tool_call():
    mock_tokenizer = _make_mock_tokenizer()
    tool_parser: ToolParser = ToolParserManager.get_tool_parser(
        "glm47")(mock_tokenizer)

    model_deltas = [
        "I will call the tool now.\n",
        "<tool_call>",
        "\n",
        "bash\n",
        "<arg_key>command</arg_key><arg_value>echo TOOL_OK</arg_value>\n",
        "</tool_call>",
    ]

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_deltas,
                                              streaming=True)

    assert content == "I will call the tool now.\n"
    assert tool_calls == [make_tool_call("bash", {"command": "echo TOOL_OK"})]

