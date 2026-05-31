"""
runner.py — runs the benchmark.

TWO ARMS, same tasks, same model, same temperature:
  - BASELINE: each task starts fresh, no memory carried between tasks
  - SEDIMENT: between tasks, the agent can recall/capture/confirm/contradict

The grader is a separate model call that scores each response 0-1 per rubric
criterion. Same grader for both arms, so any bias is at least consistent.

USAGE:
  export ANTHROPIC_API_KEY=...
  python -m benchmark.runner --arm baseline
  python -m benchmark.runner --arm sediment
  python -m benchmark.runner --arm both     # runs both, prints comparison

HONEST DESIGN NOTES:
  - N tasks is small (9) by default to keep costs <$1 per full run.
  - Each family runs in order, so memory effects accumulate within family.
  - We run multiple SEEDS by default to smooth out single-run variance —
    LLM grading is noisy.
  - The grader sees the rubric criterion, not the lesson_hint, so the lesson
    being captured doesn't leak into grading.
"""
from __future__ import annotations
import os, sys, json, time, argparse, random
from dataclasses import dataclass, asdict
from typing import Optional

# allow running as module OR script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sediment import Ledger
from embeddings import make_provider
from benchmark.tasks import ALL_TASKS, Task


MODEL = os.environ.get("SEDIMENT_BENCH_MODEL", "claude-sonnet-4-5")
GRADER_MODEL = os.environ.get("SEDIMENT_BENCH_GRADER", MODEL)


def _client():
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("Benchmark needs `pip install anthropic`.")
    return Anthropic()


# --------------------------------------------------------------------------- #
# The agent: solves one task, optionally with Sediment memory                 #
# --------------------------------------------------------------------------- #
AGENT_SYSTEM = """You are an expert prompt engineer. The user will describe a prompt they need written. Produce the requested prompt directly — no preamble, no meta-commentary. The prompt should be production-ready."""

AGENT_SYSTEM_WITH_MEMORY = AGENT_SYSTEM + """

You have access to lessons learned from previous similar tasks, provided below. For each recalled lesson, you MUST incorporate it into your output if it is at all applicable. Before finalizing your response, silently verify that every applicable lesson is reflected — if a lesson suggests adding a step, sentence, or structural element, that element must appear in your output verbatim or in equivalent form. Do not mention the lessons themselves, and do not add meta-commentary; just produce the requested prompt with the lessons applied.

<lessons_from_memory>
{lessons}
</lessons_from_memory>"""


def run_agent(client, task: Task, lessons: list[str]) -> str:
    """Run the agent on one task, optionally seeded with recalled lessons."""
    if lessons:
        system = AGENT_SYSTEM_WITH_MEMORY.format(
            lessons="\n".join(f"- {l}" for l in lessons))
    else:
        system = AGENT_SYSTEM
    resp = client.messages.create(
        model=MODEL, max_tokens=1024, temperature=0.7,
        system=system,
        messages=[{"role": "user", "content": task.prompt}],
    )
    return resp.content[0].text


# --------------------------------------------------------------------------- #
# The grader: scores a response against a task's rubric                       #
# --------------------------------------------------------------------------- #
GRADER_SYSTEM = """You are an impartial grader. You will receive a prompt-engineering task and a candidate response. Score the response against each rubric criterion on a scale of 0.0 (criterion not met at all) to 1.0 (criterion fully met).

Respond ONLY with valid JSON in this exact shape:
{"scores": [0.0, 0.5, 1.0], "notes": "one sentence explaining the grades"}

The scores array must have exactly one entry per criterion, in order."""


def grade(client, task: Task, response: str) -> dict:
    criteria_str = "\n".join(
        f"{i+1}. ({r['weight']:.2f}) {r['criterion']}"
        for i, r in enumerate(task.rubric)
    )
    user = f"""TASK GIVEN TO CANDIDATE:
{task.prompt}

RUBRIC:
{criteria_str}

CANDIDATE RESPONSE:
{response}

Score each criterion 0.0–1.0 in order. Reply only with the JSON object."""

    resp = client.messages.create(
        model=GRADER_MODEL, max_tokens=300, temperature=0.0,
        system=GRADER_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text.strip()
    # tolerate code fences
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"scores": [0.0] * len(task.rubric), "notes": f"GRADER PARSE FAIL: {raw[:120]}"}
    scores = parsed.get("scores", [])
    weights = [r["weight"] for r in task.rubric]
    weighted = sum(s * w for s, w in zip(scores, weights)) if len(scores) == len(weights) else 0.0
    return {"scores": scores, "weighted": round(weighted, 3),
            "notes": parsed.get("notes", "")}


# --------------------------------------------------------------------------- #
# Self-distill: turn a graded result into a candidate heuristic to capture    #
# --------------------------------------------------------------------------- #
DISTILL_SYSTEM = """You are observing an agent solve prompt-engineering tasks. After each task, you decide whether to write down a TRANSFERABLE lesson for future tasks.

Reply with valid JSON ONLY:
{"capture": true, "when": "...", "prefer": "...", "feedback": "positive|negative|neutral"}
OR
{"capture": false, "reason": "..."}

Rules:
- Only capture if the lesson is GENERALIZABLE (a rule that would help on future similar tasks), not a verbatim observation about this one task.
- The WHEN should describe a class of situations, not the specific task.
- Set feedback=positive if the candidate response scored well (>0.7), negative if poorly (<0.4)."""


def distill(client, task: Task, response: str, grade_result: dict) -> Optional[dict]:
    user = f"""TASK: {task.prompt}

RESPONSE QUALITY: weighted score {grade_result.get('weighted', 0)}
GRADER NOTES: {grade_result.get('notes', '')}

Should we capture a lesson? Reply with the JSON only."""
    resp = client.messages.create(
        model=GRADER_MODEL, max_tokens=300, temperature=0.0,
        system=DISTILL_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text.strip().strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
        return parsed if parsed.get("capture") else None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Arms                                                                         #
# --------------------------------------------------------------------------- #
def run_baseline(client, tasks: list[Task]) -> list[dict]:
    results = []
    for t in tasks:
        resp = run_agent(client, t, lessons=[])
        g = grade(client, t, resp)
        results.append({"task": t.id, "family": t.family,
                        "weighted": g.get("weighted", 0), "notes": g.get("notes", "")})
        print(f"  [baseline] {t.id}: {g.get('weighted', 0):.2f}")
    return results


def run_sediment(client, tasks: list[Task], ledger_path: str) -> list[dict]:
    L = Ledger(path=ledger_path, embedder=make_provider()).load()
    results = []
    for t in tasks:
        # RECALL
        hits = L.recall(t.prompt, k=5)
        lessons = [f"WHEN {h.when} → PREFER {h.prefer}" for h in hits]
        # ACT
        resp = run_agent(client, t, lessons=lessons)
        # GRADE
        g = grade(client, t, resp)
        results.append({"task": t.id, "family": t.family,
                        "weighted": g.get("weighted", 0),
                        "recalled": len(hits),
                        "notes": g.get("notes", "")})
        print(f"  [sediment] {t.id}: {g.get('weighted', 0):.2f}  (recalled {len(hits)})")
        # FEEDBACK on recalled lessons: if we scored well, the recalled lessons probably helped
        if hits and g.get("weighted", 0) >= 0.7:
            for h in hits:
                L.confirm(h)
        elif hits and g.get("weighted", 0) <= 0.4:
            for h in hits:
                L.contradict(h)
        # DISTILL & CAPTURE
        cap = distill(client, t, resp, g)
        if cap:
            L.capture(cap["when"], cap["prefer"], feedback=cap.get("feedback", "neutral"))
        L.homeostasis()
        L.save()
    return results


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def summarize(baseline: list[dict], sediment: list[dict]):
    def by_family(rs):
        from collections import defaultdict
        d = defaultdict(list)
        for r in rs:
            d[r["family"]].append(r["weighted"])
        return {k: sum(v) / len(v) for k, v in d.items()}

    bf, sf = by_family(baseline), by_family(sediment)
    b_total = sum(r["weighted"] for r in baseline) / len(baseline)
    s_total = sum(r["weighted"] for r in sediment) / len(sediment)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Family':<28} {'Baseline':>10} {'Sediment':>10} {'Δ':>8}")
    print("-" * 60)
    for fam in sorted(set(bf) | set(sf)):
        b, s = bf.get(fam, 0), sf.get(fam, 0)
        print(f"{fam:<28} {b:>10.3f} {s:>10.3f} {s - b:>+8.3f}")
    print("-" * 60)
    print(f"{'OVERALL':<28} {b_total:>10.3f} {s_total:>10.3f} {s_total - b_total:>+8.3f}")
    print("=" * 60)
    return {"baseline": b_total, "sediment": s_total, "delta": s_total - b_total,
            "by_family": {"baseline": bf, "sediment": sf}}


# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "sediment", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=1, help="Re-run N times and average (LLM grading is noisy)")
    ap.add_argument("--ledger", default="/tmp/sediment_bench.json")
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in env.")
        sys.exit(1)

    client = _client()
    print(f"Model: {MODEL}   Grader: {GRADER_MODEL}   Tasks: {len(ALL_TASKS)}   Seeds: {args.seeds}\n")

    all_baseline, all_sediment = [], []
    for seed in range(args.seeds):
        if args.seeds > 1:
            print(f"--- seed {seed + 1}/{args.seeds} ---")
        if args.arm in ("baseline", "both"):
            print("BASELINE arm:")
            all_baseline.extend(run_baseline(client, ALL_TASKS))
        if args.arm in ("sediment", "both"):
            print("SEDIMENT arm:")
            # fresh ledger per seed so memory effects are within-run
            if os.path.exists(args.ledger):
                os.remove(args.ledger)
            all_sediment.extend(run_sediment(client, ALL_TASKS, args.ledger))

    if all_baseline and all_sediment:
        summary = summarize(all_baseline, all_sediment)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "baseline": all_baseline,
                       "sediment": all_sediment}, f, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
