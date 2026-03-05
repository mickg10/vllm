# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning import ReasoningParserManager


def test_glm_reasoning_parser_aliases_registered():
    # The GLM-4.7 model card expects `--reasoning-parser glm45`. Ensure the
    # alias exists in the TT fork.
    assert "glm4_moe" in ReasoningParserManager.reasoning_parsers
    assert "glm45" in ReasoningParserManager.reasoning_parsers

    assert (ReasoningParserManager.get_reasoning_parser("glm45")
            is ReasoningParserManager.get_reasoning_parser("glm4_moe"))

