# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.tool_parsers import ToolParserManager


def test_glm_tool_parser_aliases_registered():
    # The GLM-4.7 model card expects `--tool-call-parser glm47` (and sometimes
    # `glm45`). Ensure these aliases exist in the TT fork.
    assert "glm4_moe" in ToolParserManager.tool_parsers
    assert "glm45" in ToolParserManager.tool_parsers
    assert "glm47" in ToolParserManager.tool_parsers

    assert (ToolParserManager.get_tool_parser("glm47")
            is ToolParserManager.get_tool_parser("glm4_moe"))
    assert (ToolParserManager.get_tool_parser("glm45")
            is ToolParserManager.get_tool_parser("glm4_moe"))

