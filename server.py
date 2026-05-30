"""
sediment_mcp — an MCP server exposing the Sediment memory-consolidation engine
as live tools, so any MCP-capable agent can RECALL lessons, CAPTURE new ones,
and CONSOLIDATE its memory the way the brain does (decay + sleep).

The ledger persists to a JSON file (env SEDIMENT_LEDGER_PATH, default
~/.sediment/ledger.json). Every tool loads → mutates → saves, so the memory is
externalized and durable across runs and processes — never held only in context.

Run locally (stdio):     python server.py
Run remotely (HTTP):     python server.py --http   (or set SEDIMENT_HTTP=1)
Inspect:                 npx @modelcontextprotocol/inspector python server.py
"""

from __future__ import annotations
import os, sys, json
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

from sediment import Ledger, Heuristic, text_sim

# --------------------------------------------------------------------------- #
# Configuration & shared helpers                                              #
# --------------------------------------------------------------------------- #
LEDGER_PATH = os.environ.get(
    "SEDIMENT_LEDGER_PATH",
    os.path.join(os.path.expanduser("~"), ".sediment", "ledger.json"),
)

mcp = FastMCP("sediment_mcp")


def _ledger() -> Ledger:
    """Load the ledger fresh each call (externalized, process-safe)."""
    os.makedirs(os.path.dirname(LEDGER_PATH) or ".", exist_ok=True)
    return Ledger(path=LEDGER_PATH).load()


def _entry_view(e: Heuristic) -> dict:
    return {
        "id": e.id,
        "when": e.when,
        "prefer": e.prefer,
        "confidence": round(e.effective_confidence(), 3),
        "hits": e.hit_count,
        "confirmations": e.confirm_count,
        "status": e.status,
        "note": e.note,
    }


def _ok(payload: dict) -> str:
    return json.dumps({"ok": True, **payload}, indent=2)


def _err(msg: str) -> str:
    return json.dumps({"ok": False, "error": msg}, indent=2)


class Feedback(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# --------------------------------------------------------------------------- #
# RECALL                                                                      #
# --------------------------------------------------------------------------- #
class RecallInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    context: str = Field(..., min_length=1, max_length=2000,
                         description="Description of the current task, e.g. 'build a Pine Script for XAUUSD'")
    k: int = Field(default=5, ge=1, le=20,
                   description="Max heuristics to return, ranked by relevance x confidence")


@mcp.tool(
    name="sediment_recall",
    annotations={"title": "Recall relevant lessons", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def sediment_recall(params: RecallInput) -> str:
    """Load the top-K learned heuristics relevant to the current task.

    Call this at the START of a task. Returned heuristics should inform the work;
    note in your output which ones fired. Recall also increments each hit's usage,
    which keeps frequently-useful lessons alive during consolidation.

    Returns JSON: {ok, count, heuristics:[{id, when, prefer, confidence, hits,
    confirmations, status, note}]}.
    """
    L = _ledger()
    hits = L.recall(params.context, k=params.k)
    L.save()
    return _ok({"count": len(hits), "heuristics": [_entry_view(e) for e in hits]})


# --------------------------------------------------------------------------- #
# CAPTURE                                                                      #
# --------------------------------------------------------------------------- #
class CaptureInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    when: str = Field(..., min_length=1, max_length=300,
                      description="The TRIGGER condition, generalized — e.g. 'non-reasoning GPT target'")
    prefer: str = Field(..., min_length=1, max_length=300,
                        description="The ACTION to take when the trigger holds — e.g. 'inject explicit CoT'")
    feedback: Feedback = Field(default=Feedback.NEUTRAL,
                               description="Outcome signal: 'positive' if the lesson was confirmed by a good result, "
                                           "'negative' if it failed, 'neutral' if untested")


@mcp.tool(
    name="sediment_capture",
    annotations={"title": "Capture a new lesson", "readOnlyHint": False,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def sediment_capture(params: CaptureInput) -> str:
    """Distill an episode into a reusable, generalized heuristic and store it.

    Call this at the END of a task when there is a REAL outcome signal. Write the
    trigger/action as a transferable rule ('WHEN x -> PREFER y'), never a verbatim
    log of one event. No feedback signal -> capture as neutral (weak, prunable).

    Returns JSON: {ok, captured:{id, when, prefer, confidence, ...}}.
    """
    L = _ledger()
    h = L.capture(params.when, params.prefer, feedback=params.feedback.value)
    L.save()
    return _ok({"captured": _entry_view(h)})


# --------------------------------------------------------------------------- #
# CONFIRM / CONTRADICT (reconsolidation signals)                              #
# --------------------------------------------------------------------------- #
class IdInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    id: str = Field(..., min_length=1, max_length=32,
                    description="The heuristic id (from sediment_recall / sediment_stats)")


@mcp.tool(
    name="sediment_confirm",
    annotations={"title": "Confirm a lesson worked", "readOnlyHint": False,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def sediment_confirm(params: IdInput) -> str:
    """Reconsolidate: a recalled lesson was applied and WORKED. Strengthens its
    confidence and lengthens its memory half-life (spaced repetition).

    Returns JSON: {ok, updated:{...}} or an error if the id is unknown.
    """
    L = _ledger()
    h = L.get(params.id)
    if not h:
        return _err(f"No heuristic with id '{params.id}'. Call sediment_stats to list ids.")
    L.confirm(h)
    L.save()
    return _ok({"updated": _entry_view(h)})


@mcp.tool(
    name="sediment_contradict",
    annotations={"title": "Flag a lesson that failed", "readOnlyHint": False,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def sediment_contradict(params: IdInput) -> str:
    """A recalled lesson was applied and FAILED. Halves its confidence; repeated
    contradiction lets consolidation prune it. Use this instead of deleting —
    the decay path keeps an audit trail.

    Returns JSON: {ok, updated:{...}} or an error if the id is unknown.
    """
    L = _ledger()
    h = L.get(params.id)
    if not h:
        return _err(f"No heuristic with id '{params.id}'. Call sediment_stats to list ids.")
    L.contradict(h)
    L.save()
    return _ok({"updated": _entry_view(h)})


# --------------------------------------------------------------------------- #
# CONSOLIDATE (cheap homeostasis)                                             #
# --------------------------------------------------------------------------- #
@mcp.tool(
    name="sediment_consolidate",
    annotations={"title": "Run cheap memory hygiene", "readOnlyHint": False,
                 "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
async def sediment_consolidate() -> str:
    """Run the CHEAP homeostasis pass (safe to call every run): apply the
    forgetting curve, retire faded lessons, and evict the lowest-value entries if
    over the budget cap. Does NOT merge or resolve contradictions — that is sleep.

    Returns JSON: {ok, retired, evicted, active}.
    """
    L = _ledger()
    stats = L.homeostasis()
    L.save()
    return _ok(stats)


# --------------------------------------------------------------------------- #
# SLEEP (deep consolidation, deterministic one-shot)                          #
# --------------------------------------------------------------------------- #
@mcp.tool(
    name="sediment_sleep",
    annotations={"title": "Deep consolidation (deterministic)", "readOnlyHint": False,
                 "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
async def sediment_sleep() -> str:
    """Run the EXPENSIVE consolidation pass using the built-in lexical judge:
    merge duplicates, resolve contradictions into boundary rules, abstract sibling
    clusters into parents. Run periodically, not every turn.

    For higher-quality semantic judgment, prefer the review round-trip
    (sediment_review_candidates -> sediment_apply_judgments), which lets YOU, the
    agent, be the judge.

    Returns JSON: {ok, merged, contradictions_resolved, generalized}.
    """
    L = _ledger()
    stats = L.sleep()  # lexical fallback
    L.save()
    return _ok(stats)


# --------------------------------------------------------------------------- #
# AGENT-AS-JUDGE round-trip: review candidates, then apply judgments          #
# --------------------------------------------------------------------------- #
@mcp.tool(
    name="sediment_review_candidates",
    annotations={"title": "List pairs needing a judgment", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def sediment_review_candidates() -> str:
    """Return pairs of active heuristics similar enough to need a semantic verdict.
    YOU decide each: 'merge' (same lesson, paraphrased), 'contradict' (same trigger,
    opposing action), or 'distinct'. Then send your verdicts to
    sediment_apply_judgments. This makes the agent the semantic judge — the part a
    deterministic engine cannot do well.

    Returns JSON: {ok, count, pairs:[{a_id, b_id, a, b, trigger_similarity,
    action_similarity}]}.
    """
    L = _ledger()
    pairs = L.candidate_pairs()
    return _ok({"count": len(pairs), "pairs": pairs})


class Judgment(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    a_id: str = Field(..., description="First heuristic id from the candidate pair")
    b_id: str = Field(..., description="Second heuristic id from the candidate pair")
    verdict: str = Field(..., pattern="^(merge|contradict|distinct)$",
                         description="'merge', 'contradict', or 'distinct'")


class ApplyJudgmentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    judgments: List[Judgment] = Field(..., min_length=1, max_length=100,
                                      description="Verdicts for pairs from sediment_review_candidates")


@mcp.tool(
    name="sediment_apply_judgments",
    annotations={"title": "Apply your merge/contradict verdicts", "readOnlyHint": False,
                 "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
async def sediment_apply_judgments(params: ApplyJudgmentsInput) -> str:
    """Apply the verdicts you produced from sediment_review_candidates. 'merge'
    folds the pair into one canonical lesson; 'contradict' keeps the stronger one
    and records the weaker as a boundary exception; 'distinct' is a no-op.

    Returns JSON: {ok, merged, contradictions_resolved, skipped}.
    """
    L = _ledger()
    merged = resolved = skipped = 0
    for j in params.judgments:
        if j.verdict == "merge":
            merged += 1 if L.apply_merge(j.a_id, j.b_id) else 0
            skipped += 0 if L.get(j.a_id) else 1
        elif j.verdict == "contradict":
            resolved += 1 if L.apply_contradiction(j.a_id, j.b_id) else 0
        else:
            skipped += 1
    L.save()
    return _ok({"merged": merged, "contradictions_resolved": resolved, "skipped": skipped})


# --------------------------------------------------------------------------- #
# STATS / inspection                                                          #
# --------------------------------------------------------------------------- #
class StatsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_retired: bool = Field(default=False, description="Include retired/dormant entries")
    limit: int = Field(default=50, ge=1, le=500, description="Max entries to return")


@mcp.tool(
    name="sediment_stats",
    annotations={"title": "Inspect the ledger", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def sediment_stats(params: StatsInput) -> str:
    """Inspect ledger state: counts and the current entries (ids, triggers, actions,
    confidence). Use to find ids for confirm/contradict or to audit memory health.

    Returns JSON: {ok, path, total, active, entries:[...]}.
    """
    L = _ledger()
    pool = L.entries if params.include_retired else L.active
    pool = sorted(pool, key=lambda e: e.effective_confidence(), reverse=True)[:params.limit]
    return _ok({"path": LEDGER_PATH, "total": len(L.entries), "active": len(L.active),
                "entries": [_entry_view(e) for e in pool]})


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if "--http" in sys.argv or os.environ.get("SEDIMENT_HTTP") == "1":
        mcp.run(transport="streamable_http")
    else:
        mcp.run()  # stdio (default)
