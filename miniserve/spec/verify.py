"""The acceptance rule: which of a round's proposals the target model keeps.

A verify pass runs ``[last token, p_0, ..., p_{g-1}]`` through the target in one
forward, so it produces what the target would choose at each of those ``g + 1``
positions *given that every earlier proposal was accepted*. Comparing those
choices with the proposals from left to right is therefore exactly what
one-token-at-a-time decoding would have done, up to the first disagreement.

Greedy decoding keeps the longest agreeing prefix and then the target's own
choice at the first position where they disagree (the *bonus* token, which the
verify pass computed anyway). So a round always produces at least one token and
at most ``g + 1``, and the tokens it produces are the ones plain decoding would
have produced: speculation changes the speed, never the output.
"""

from __future__ import annotations

from collections.abc import Sequence


def accept_prefix(proposals: Sequence[int], chosen: Sequence[int]) -> int:
    """How many proposals the target agrees with, from the left.

    ``chosen[i]`` is the target's token for the position ``proposals[i]`` occupies;
    ``chosen`` has one more entry than ``proposals`` (the bonus position).
    """
    if len(chosen) != len(proposals) + 1:
        raise ValueError(f"{len(chosen)} verified positions for {len(proposals)} proposals")
    for i, p in enumerate(proposals):
        if chosen[i] != p:
            return i
    return len(proposals)


def accepted_tokens(proposals: Sequence[int], chosen: Sequence[int]) -> list[int]:
    """The tokens a round produces: the agreeing prefix plus the target's next token."""
    k = accept_prefix(proposals, chosen)
    return [*proposals[:k], chosen[k]]
