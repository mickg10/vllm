# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side rejection sampler for TT speculative decoding.

TT devices cannot run Triton kernels, so rejection sampling runs on CPU.
Currently supports greedy (temperature=0) only. Random sampling support
is a future extension.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TTRejectionResult:
    """Per-request rejection sampling result."""
    # Per-request: number of accepted draft tokens (0 to K).
    num_accepted: list[int]
    # Per-request: list of output token IDs.
    # Length = num_accepted + 1 (accepted drafts + bonus token).
    output_token_ids: list[np.ndarray]


class TTRejectionSampler:
    """Host-side greedy rejection sampler for TT spec decode.

    For K=1 approximate verification:
    - main_logits[b] = target model logits at position P for request b
    - draft_logits[b] = target model logits at position P+1 for request b
    - draft_token_ids[b] = the draft token proposed for position P+1

    Correct K=1 flow:
    - Previous step: MTP produced draft_token for position P+1
    - Current step: main model runs at P (main) and P+1 (draft verification)
    - main_logits at P -> argmax = sampled_main (the "real" next token)
    - draft_logits at P+1 -> argmax = bonus_token (what model predicts at P+2)
    - Accept draft if: sampled_main == draft_token_ids[b][0]
    - If accepted: output = [sampled_main, bonus_token] (2 tokens)
    - If rejected: output = [sampled_main] (1 token, draft discarded)
    """

    def __call__(
        self,
        draft_token_ids: list[list[int]],  # [B, K] draft tokens per request
        main_logits: torch.Tensor,          # [B, vocab] logits at main positions
        draft_logits: torch.Tensor,         # [B, vocab] logits at draft positions
    ) -> TTRejectionResult:
        """Run greedy rejection sampling.

        Args:
            draft_token_ids: Per-request draft token IDs from MTP. For K=1,
                each inner list has exactly 1 element.
            main_logits: Target model logits at the main token position (P).
                Shape [B, vocab_size].
            draft_logits: Target model logits at the draft token position (P+1).
                Shape [B, vocab_size].

        Returns:
            TTRejectionResult with accepted counts and output token lists.
        """
        batch = len(draft_token_ids)
        num_accepted = []
        output_token_ids = []

        # Batch argmax for efficiency
        main_sampled = main_logits.argmax(dim=-1)   # [B]
        bonus_tokens = draft_logits.argmax(dim=-1)   # [B]

        for b in range(batch):
            sampled_main = int(main_sampled[b].item())
            K = len(draft_token_ids[b])
            accepted = 0

            # For K=1: check if the draft matches the main model's choice
            if K > 0 and sampled_main == draft_token_ids[b][0]:
                accepted = 1

            if accepted > 0:
                # Draft accepted: output = [main_token, bonus_token]
                bonus = int(bonus_tokens[b].item())
                tokens = np.array(
                    [sampled_main, bonus], dtype=np.int32
                )
            else:
                # Draft rejected: output = [main_token] only
                tokens = np.array([sampled_main], dtype=np.int32)

            num_accepted.append(accepted)
            output_token_ids.append(tokens)

        return TTRejectionResult(
            num_accepted=num_accepted,
            output_token_ids=output_token_ids,
        )
