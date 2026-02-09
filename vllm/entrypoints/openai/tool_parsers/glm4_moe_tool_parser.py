# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# code modified from deepseekv3_tool_parser.py

from collections.abc import Sequence
from typing import Union

import regex as re

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


# The GLM model card recommends `--tool-call-parser glm47` (and sometimes
# `glm45`). Keep backwards compatibility with the existing `glm4_moe` name.
@ToolParserManager.register_module(["glm4_moe", "glm45", "glm47"])
class Glm4MoeModelToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        self.tool_calls_start_token = self.tool_call_start_token

        # Updated regex for the XML-based format
        self.tool_call_regex = re.compile(
            r"<tool_call>\s*"
            r"(?P<function_name>[^\n<]+)\s*"  # 函数名（到换行或 <）
            r"(?P<arguments>(?:\s*<arg_key>[^<]+</arg_key>\s*"
            r"<arg_value>[^<]*</arg_value>\s*)*)\s*"
            r"</tool_call>",
            re.DOTALL,
        )

        # Regex for parsing individual arguments
        self.arg_regex = re.compile(
            r"<arg_key>(?P<key>[^<]+)</arg_key>\s*<arg_value>(?P<value>[^<]*)</arg_value>",
            re.DOTALL,
        )

        # Streaming regex
        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<function_name>[^\n<]+)\s*"
            r"(?P<arguments>(?:\s*<arg_key>[^<]+</arg_key>\s*"
            r"<arg_value>[^<]*</arg_value>\s*)*)",
            re.DOTALL,
        )

        # For streaming, we also need a regex to match just the function name
        self.stream_tool_call_name_regex = re.compile(
            r"(?P<function_name>[^\n<]+)",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        if (self.tool_call_start_token_id is None
                or self.tool_call_end_token_id is None):
            raise RuntimeError(
                "GLM4 MoE tool parser could not locate tool call start/end "
                "tokens in the tokenizer!")

        # Streaming state for XML tool calls.
        self._streaming_in_tool_call: bool = False
        self._streaming_tool_call_buffer: str = ""

    def _parse_arguments_dict(self, args_text: str) -> dict[str, str]:
        """Parse XML-based arguments into a simple key/value dict."""
        if not args_text or not args_text.strip():
            return {}

        args: dict[str, str] = {}
        for key, value in self.arg_regex.findall(args_text):
            args[key.strip()] = value.strip()
        return args

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        # sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            # Find all tool calls in the output
            function_call_matches = self.tool_call_regex.findall(model_output)

            logger.debug("function_call_matches: %s", function_call_matches)

            if not function_call_matches:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            tool_calls = []
            for i, match in enumerate(function_call_matches):
                function_name, function_args_xml = match
                function_name = function_name.strip()

                # Parse XML arguments to a JSON string (OpenAI-compatible).
                import json
                function_args_dict = self._parse_arguments_dict(
                    function_args_xml)
                function_args_json = json.dumps(function_args_dict,
                                                ensure_ascii=False)

                tool_calls.append(
                    ToolCall(
                        id=f"call_{i}",
                        type='function',
                        function=FunctionCall(name=function_name,
                                              arguments=function_args_json),
                    ))

            # Extract content before the first tool call
            content = model_output[:model_output.find(self.
                                                      tool_calls_start_token)]
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content.strip() if content.strip() else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        # We keep streaming logic intentionally simple for the GLM XML tool
        # format:
        # 1) Stream normal text until we see `<tool_call>`.
        # 2) Buffer tool-call text until `</tool_call>`.
        # 3) Emit a single tool-call delta with JSON arguments.
        #
        # This matches the OpenAI streaming contract well enough for tool-using
        # clients (including opencode), while avoiding partial-JSON
        # autocompletion logic that doesn't apply to XML arguments.

        import json

        if not self._streaming_in_tool_call:
            if self.tool_call_start_token not in delta_text:
                return DeltaMessage(content=delta_text)

            # Split any prefix content before the tool call start token.
            prefix, after = delta_text.split(self.tool_call_start_token, 1)

            # Enter tool-call buffering state.
            self._streaming_in_tool_call = True
            self._streaming_tool_call_buffer = after

            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")
            # Placeholder to indicate tools were called (used by serving code).
            self.prev_tool_call_arr.append({})

            return DeltaMessage(content=prefix) if prefix else None

        # Buffering tool-call content (start token already consumed).
        self._streaming_tool_call_buffer += delta_text

        if self.tool_call_end_token not in self._streaming_tool_call_buffer:
            return None

        tool_text, _rest = self._streaming_tool_call_buffer.split(
            self.tool_call_end_token, 1)
        self._streaming_tool_call_buffer = ""
        self._streaming_in_tool_call = False

        # Parse the tool name + XML arguments.
        m = self.stream_tool_call_portion_regex.match(tool_text.strip())
        if not m:
            logger.debug("Unable to parse GLM tool call payload: %r", tool_text)
            return None

        tool_name, args_xml = m.groups()
        tool_name = tool_name.strip()
        args_dict = self._parse_arguments_dict(args_xml)
        args_json = json.dumps(args_dict, ensure_ascii=False)

        # Update serving-code state so finish_reason becomes `tool_calls`.
        tool_id = f"call_{self.current_tool_id}"
        self.prev_tool_call_arr[self.current_tool_id] = {
            "id": tool_id,
            "name": tool_name,
            "arguments": args_dict,
        }
        self.streamed_args_for_tool[self.current_tool_id] = args_json

        self.current_tool_name_sent = True
        return DeltaMessage(tool_calls=[
            DeltaToolCall(
                index=self.current_tool_id,
                type="function",
                id=tool_id,
                function=DeltaFunctionCall(name=tool_name,
                                           arguments=args_json).model_dump(
                                               exclude_none=True),
            )
        ])
