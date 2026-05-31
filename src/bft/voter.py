"""
Deterministic BFT quorum certifier.

After all generator nodes have emitted a :class:`BFTGeneratorOutput`, the
voter is responsible for selecting the certified code that should proceed
downstream. Selection rules:

  1. Discard every output with ``status="FAILED"`` and every output whose
     ``ast_fingerprint`` is missing or equal to ``SYNTAX_ERROR_FINGERPRINT``.
  2. Tally the remaining fingerprints with :class:`collections.Counter`.
  3. The most common fingerprint must appear at least ``Q = 2f + 1`` times.
     Otherwise the kill switch :exc:`BFTQuorumFailure` fires (KS-1).
  4. On success, pick *any* OK output whose fingerprint matches the winner
     and use its ``raw_code`` as the certified output. Selection among ties
     is deterministic — we pick the lowest ``node_id``.
"""

from __future__ import annotations

from collections import Counter

from loguru import logger

from .ast_normalizer import SYNTAX_ERROR_FINGERPRINT
from .models import (
    BFTConsensusResult,
    BFTGeneratorOutput,
    BFTQuorumFailure,
)


async def certify_quorum(
    outputs: list[BFTGeneratorOutput],
    f: int,
) -> BFTConsensusResult:
    """Apply BFT quorum certification over generator fingerprints.

    Parameters
    ----------
    outputs:
        List of generator outputs (length should be ``N`` but the voter
        tolerates fewer if some generators were entirely lost).
    f:
        Maximum Byzantine faults tolerated. Quorum threshold ``Q = 2*f + 1``.

    Raises
    ------
    BFTQuorumFailure
        Kill switch KS-1 — no fingerprint achieved the required quorum.
    """

    quorum_threshold: int = 2 * f + 1
    total_nodes: int = len(outputs)

    # ── Step 1: filter invalid contributions ────────────────────────────────
    valid: list[BFTGeneratorOutput] = [
        o
        for o in outputs
        if o.status == "OK"
        and o.ast_fingerprint
        and o.ast_fingerprint != SYNTAX_ERROR_FINGERPRINT
    ]

    # ── Step 2: tally fingerprints ──────────────────────────────────────────
    fingerprints: list[str] = [o.ast_fingerprint or "" for o in valid]
    counter: Counter[str] = Counter(fingerprints)
    divergence_map: dict[str, int] = dict(counter)

    if not counter:
        logger.error(
            "BFT voter: every generator failed or produced a syntax error "
            f"(total_nodes={total_nodes}, valid=0, Q={quorum_threshold})"
        )
        raise BFTQuorumFailure(
            "ABORT: quorum not achieved — zero valid generator outputs"
        )

    winning_fingerprint, winning_count = counter.most_common(1)[0]

    logger.info(
        "BFT voter: counter={c} winner_count={w} quorum_required={q}",
        c=dict(counter),
        w=winning_count,
        q=quorum_threshold,
    )

    # ── Step 3: enforce quorum (KS-1) ───────────────────────────────────────
    if winning_count < quorum_threshold:
        logger.error(
            "BFT voter: KS-1 fired — winning fingerprint has {w} votes, "
            "need {q}. Divergence map: {d}",
            w=winning_count,
            q=quorum_threshold,
            d=divergence_map,
        )
        for o in outputs:
            logger.error(
                "  node_id={nid} status={st} fingerprint={fp!s:.120}",
                nid=o.node_id,
                st=o.status,
                fp=o.ast_fingerprint,
            )
        raise BFTQuorumFailure(
            f"ABORT: quorum not achieved (got {winning_count}/{quorum_threshold})"
        )

    # ── Step 4: pick certified code deterministically ───────────────────────
    winners = sorted(
        (o for o in valid if o.ast_fingerprint == winning_fingerprint),
        key=lambda o: o.node_id,
    )
    certified_code: str = winners[0].raw_code

    return BFTConsensusResult(
        certified_code=certified_code,
        winning_fingerprint=winning_fingerprint,
        quorum_count=winning_count,
        total_nodes=total_nodes,
        divergence_map=divergence_map,
        quorum_achieved=True,
    )


__all__: list[str] = ["certify_quorum"]
