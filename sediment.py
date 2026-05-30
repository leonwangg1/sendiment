"""
Sediment — a memory-consolidation engine for self-improving agents.

The problem: an agent that appends every lesson to a ledger rots. It fills with
near-duplicates, holds contradictory rules side by side, and never forgets stale
ones — so the ledger eventually DEGRADES output instead of improving it.

The brain solves this with two processes at two rates, and so does this tool:

  1. SYNAPTIC HOMEOSTASIS  (cheap, every run)
     - time-decay of confidence (a forgetting curve)
     - reconsolidation on confirmation (spaced repetition lengthens retention)
     - retirement of traces that fall below a floor, eviction under a budget cap

  2. SYSTEMS CONSOLIDATION  ("sleep", periodic / when redundancy is high)
     - MERGE   semantically duplicate heuristics into one canonical entry
     - RESOLVE contradictions (same trigger, opposite action) — by GENERALIZING
               into a boundary rule where possible, not just deleting the loser
     - ABSTRACT clusters of specific siblings into a parent heuristic

The deterministic core needs no dependencies and no network, so it runs anywhere.
The semantic judgments (is-same? / contradicts? / generalize) are pluggable hooks:
pass a real `judge` / `summarizer` (an LLM call) for production quality; omit them
and the engine falls back to defensible lexical heuristics.
"""

from __future__ import annotations
import json, math, time, uuid, re
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

NOW = lambda: time.time()
DAY = 86400.0


# --------------------------------------------------------------------------- #
# Similarity (dependency-free). Swap `text_sim` for real embeddings in prod.   #
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9]+")

def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))

def _trigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return {s[i:i+3] for i in range(len(s) - 2)} or {s}

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

def text_sim(a: str, b: str) -> float:
    """Hybrid lexical similarity in [0,1]. Lexical is a stand-in for embeddings."""
    return 0.5 * _jaccard(_tokens(a), _tokens(b)) + 0.5 * _jaccard(_trigrams(a), _trigrams(b))


# --------------------------------------------------------------------------- #
# The unit of memory                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Heuristic:
    when: str                       # trigger condition
    prefer: str                     # action to take
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    confidence: float = 0.30        # stored base; effective value decays from here
    created: float = field(default_factory=NOW)
    last_confirmed: float = field(default_factory=NOW)
    last_fired: float = field(default_factory=NOW)
    hit_count: int = 0              # times applied
    confirm_count: int = 0          # times confirmed by positive feedback
    contradiction_count: int = 0
    status: str = "active"          # active | dormant | retired
    note: str = ""

    # spaced repetition: each confirmation lengthens how long the memory lasts
    def half_life_days(self, base: float = 14.0, k: float = 1.0) -> float:
        return base * (1.0 + k * self.confirm_count)

    def effective_confidence(self, now: Optional[float] = None) -> float:
        now = now or NOW()
        dt_days = max(0.0, (now - self.last_confirmed) / DAY)
        decay = 0.5 ** (dt_days / self.half_life_days())
        return self.confidence * decay

    def retention_value(self, now: Optional[float] = None) -> float:
        # what we sort by when evicting under a budget cap
        ec = self.effective_confidence(now)
        return ec * (1.0 + math.log1p(self.hit_count)) * (1.0 + 0.2 * self.confirm_count)


# --------------------------------------------------------------------------- #
# The ledger + the two consolidation passes                                   #
# --------------------------------------------------------------------------- #
class Ledger:
    def __init__(self, path: str = "ledger.json",
                 floor: float = 0.05, cap: int = 200,
                 trig_tau: float = 0.45, act_tau: float = 0.45,
                 cand_tau: float = 0.18):
        self.path = path
        self.floor = floor          # retire below this effective confidence
        self.cap = cap              # max active entries
        self.trig_tau = trig_tau    # trigger-similarity: the DECISION threshold
        self.act_tau = act_tau      # action-similarity threshold
        self.cand_tau = cand_tau    # loose RECALL gate: which pairs reach the judge
        self.entries: list[Heuristic] = []

    # ---- persistence -------------------------------------------------------
    def load(self):
        try:
            with open(self.path) as f:
                self.entries = [Heuristic(**d) for d in json.load(f)]
        except (FileNotFoundError, json.JSONDecodeError):
            self.entries = []
        return self

    def save(self):
        with open(self.path, "w") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)
        return self

    @property
    def active(self):
        return [e for e in self.entries if e.status == "active"]

    # ---- RECALL: top-K relevant active heuristics --------------------------
    def recall(self, context: str, k: int = 5) -> list[Heuristic]:
        now = NOW()
        scored = [(text_sim(context, f"{e.when} {e.prefer}") * e.effective_confidence(now), e)
                  for e in self.active]
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [e for s, e in scored[:k] if s > 0.0]
        for e in hits:
            e.last_fired = now
            e.hit_count += 1
        return hits

    # ---- CAPTURE: distill an episode into a heuristic ----------------------
    def capture(self, when: str, prefer: str, feedback: str = "neutral") -> Heuristic:
        h = Heuristic(when=when, prefer=prefer)
        if feedback == "positive":
            h.confirm_count, h.confidence, h.last_confirmed = 1, 0.55, NOW()
        elif feedback == "negative":
            h.confidence = 0.10        # captured but weak; likely to be pruned
        self.entries.append(h)
        return h

    def confirm(self, h: Heuristic):
        """Reconsolidation: re-applying a confirmed lesson strengthens & re-times it."""
        h.confidence = min(1.0, h.effective_confidence() + 0.15)
        h.confirm_count += 1
        h.last_confirmed = NOW()

    def contradict(self, h: Heuristic):
        h.contradiction_count += 1
        h.confidence = h.effective_confidence() * 0.5

    # ---- public API for an EXTERNAL judge (e.g. the calling LLM agent) -----
    def get(self, hid: str) -> Optional[Heuristic]:
        return next((e for e in self.entries if e.id == hid), None)

    def candidate_pairs(self) -> list[dict]:
        """Pairs of active heuristics similar enough to deserve a judgment call.
        Returned for an external judge to label merge / contradict / distinct."""
        act, out = self.active, []
        for i in range(len(act)):
            for j in range(i + 1, len(act)):
                a, b = act[i], act[j]
                ts = text_sim(a.when, b.when)
                if ts < self.cand_tau:
                    continue
                out.append({
                    "a_id": a.id, "b_id": b.id,
                    "a": f"WHEN {a.when} -> PREFER {a.prefer}",
                    "b": f"WHEN {b.when} -> PREFER {b.prefer}",
                    "trigger_similarity": round(ts, 3),
                    "action_similarity": round(text_sim(a.prefer, b.prefer), 3),
                })
        return out

    def apply_merge(self, id_a: str, id_b: str) -> bool:
        a, b = self.get(id_a), self.get(id_b)
        if not a or not b or a.status != "active" or b.status != "active":
            return False
        self._merge(a, b)
        return True

    def apply_contradiction(self, id_a: str, id_b: str) -> bool:
        a, b = self.get(id_a), self.get(id_b)
        if not a or not b or a.status != "active" or b.status != "active":
            return False
        self._resolve_contradiction(a, b, self._lexical_summarize)
        return True

    # ====================================================================== #
    # PASS 1 — SYNAPTIC HOMEOSTASIS (cheap, run every cycle)                  #
    # ====================================================================== #
    def homeostasis(self) -> dict:
        now, retired, evicted = NOW(), 0, 0
        # forgetting: bake the decay into stored confidence, then retire the faint
        for e in self.active:
            e.confidence = e.effective_confidence(now)
            e.last_confirmed = now      # decay already applied; reset the clock
            if e.confidence < self.floor:
                e.status, e.note = "retired", "faded below floor"
                retired += 1
        # budget eviction: keep the most valuable `cap` entries
        act = sorted(self.active, key=lambda x: x.retention_value(now), reverse=True)
        for e in act[self.cap:]:
            e.status, e.note = "retired", "evicted (budget cap)"
            evicted += 1
        return {"retired": retired, "evicted": evicted, "active": len(self.active)}

    # ====================================================================== #
    # PASS 2 — SYSTEMS CONSOLIDATION ("sleep", periodic)                     #
    # ====================================================================== #
    def sleep(self,
              judge: Optional[Callable[[Heuristic, Heuristic], str]] = None,
              summarizer: Optional[Callable[[list[Heuristic]], tuple[str, str]]] = None) -> dict:
        """
        judge(a, b) -> "merge" | "contradict" | "distinct"   (LLM hook; lexical fallback)
        summarizer(group) -> (when, prefer)                  (LLM hook; lexical fallback)
        """
        judge = judge or self._lexical_judge
        summarizer = summarizer or self._lexical_summarize
        stats = {"merged": 0, "contradictions_resolved": 0, "generalized": 0}

        # 1) pairwise reconcile: merge duplicates, resolve contradictions
        changed = True
        while changed:
            changed = False
            act = self.active
            for i in range(len(act)):
                for j in range(i + 1, len(act)):
                    a, b = act[i], act[j]
                    if a.status != "active" or b.status != "active":
                        continue
                    ts = text_sim(a.when, b.when)
                    if ts < self.cand_tau:        # loose recall gate, not the decision
                        continue
                    verdict = judge(a, b)
                    if verdict == "merge":
                        self._merge(a, b); stats["merged"] += 1; changed = True
                    elif verdict == "contradict":
                        self._resolve_contradiction(a, b, summarizer)
                        stats["contradictions_resolved"] += 1; changed = True
                    if changed:
                        break
                if changed:
                    break

        # 2) abstract: clusters of >=3 similar triggers -> a parent heuristic
        stats["generalized"] += self._generalize(summarizer)
        return stats

    # ---- lexical fallbacks for the semantic hooks --------------------------
    def _lexical_judge(self, a: Heuristic, b: Heuristic) -> str:
        # conservative: the lexical fallback re-checks the trigger DECISION threshold,
        # so loosening the candidate gate can't create false merges/contradictions.
        if text_sim(a.when, b.when) < self.trig_tau:
            return "distinct"
        act_sim = text_sim(a.prefer, b.prefer)
        if act_sim >= self.act_tau:
            return "merge"                      # same trigger, same action
        if act_sim <= 0.20:
            return "contradict"                 # same trigger, opposite action
        return "distinct"

    def _lexical_summarize(self, group: list[Heuristic]) -> tuple[str, str]:
        common_when = _tokens(group[0].when)
        for g in group[1:]:
            common_when &= _tokens(g.when)
        when = " ".join(sorted(common_when)) or group[0].when
        prefer = max(group, key=lambda g: g.confirm_count).prefer
        return f"(general) {when}", prefer

    # ---- structural operations --------------------------------------------
    def _merge(self, a: Heuristic, b: Heuristic):
        keep, drop = (a, b) if a.retention_value() >= b.retention_value() else (b, a)
        keep.hit_count += drop.hit_count
        keep.confirm_count += drop.confirm_count
        keep.confidence = min(1.0, max(keep.confidence, drop.confidence) + 0.05)
        keep.last_confirmed = max(keep.last_confirmed, drop.last_confirmed)
        drop.status, drop.note = "retired", f"merged into {keep.id}"

    def _resolve_contradiction(self, a, b, summarizer):
        # winner by: confirmations, then effective confidence, then recency
        rank = lambda h: (h.confirm_count, h.effective_confidence(), h.last_confirmed)
        winner, loser = (a, b) if rank(a) >= rank(b) else (b, a)
        # GENERALIZE rather than just delete: encode the loser as a boundary exception
        winner.note = (f"boundary: usually '{winner.prefer}'; "
                       f"exception toward '{loser.prefer}' when context differs").strip()
        winner.confidence = min(1.0, winner.confidence + 0.05)
        loser.status, loser.note = "retired", f"lost contradiction to {winner.id}"

    def _generalize(self, summarizer) -> int:
        act, used, made = self.active, set(), 0
        for i, a in enumerate(act):
            if a.id in used:
                continue
            cluster = [a]
            for b in act[i + 1:]:
                if b.id in used or b.id == a.id:
                    continue
                if text_sim(a.when, b.when) >= self.trig_tau and \
                   text_sim(a.prefer, b.prefer) >= self.act_tau:
                    cluster.append(b)
            if len(cluster) >= 3:
                when, prefer = summarizer(cluster)
                parent = Heuristic(when=when, prefer=prefer,
                                   confidence=min(1.0, max(c.confidence for c in cluster) + 0.1),
                                   confirm_count=sum(c.confirm_count for c in cluster),
                                   hit_count=sum(c.hit_count for c in cluster),
                                   note="abstracted parent")
                self.entries.append(parent)
                for c in cluster:
                    c.status, c.note, used = "dormant", f"folded under {parent.id}", used | {c.id}
                made += 1
        return made


# --------------------------------------------------------------------------- #
# A pretty-printer for demos / CLI                                            #
# --------------------------------------------------------------------------- #
def show(ledger: Ledger, title: str):
    print(f"\n=== {title} ===  ({len(ledger.active)} active / {len(ledger.entries)} total)")
    for e in sorted(ledger.active, key=lambda x: x.retention_value(), reverse=True):
        print(f"  [{e.id}] conf={e.effective_confidence():.2f} hits={e.hit_count} "
              f"conf#={e.confirm_count}")
        print(f"        WHEN {e.when}")
        print(f"        PREFER {e.prefer}" + (f"   ⟂ {e.note}" if e.note else ""))
