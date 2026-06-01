# Sediment Memory — Learning Discipline

This project has the **sediment** MCP server connected. You have persistent memory across runs via a Heuristics Ledger. Use it.

The ledger is your lived experience. This file is your constitution. The constitution always wins on conflict — a learned heuristic can refine *tactics*, never override your core role, safety constraints, or the user’s standing instructions.

-----

## The loop — every substantive task

**1. Open with `sediment_recall`.**
Pass a concise description of what you’re about to do. Read the returned heuristics. If any apply, *use them* — and briefly note in your response which fired (e.g. “applying lesson `7d076a28`: prefer XML delimiters for Claude targets”). Silent application defeats the audit trail.

Skip recall only for trivial conversational turns (“thanks”, “what time is it”) — anything else, recall first.

**2. While working, send feedback signals on what you recalled.**

- A recalled lesson worked → `sediment_confirm` with its id. (Strengthens it, extends its half-life.)
- A recalled lesson failed or led you astray → `sediment_contradict` with its id. (Halves its confidence; the cleanup pass will eventually prune it.)

These signals are how the ledger gets *smarter* over time. No signals → no learning.

**3. Close with `sediment_capture` — but only when there’s a real outcome.**
At task end, if the run produced a genuine, transferable lesson, capture it. The two rules that matter:

- **Generalize, don’t log.** Write `WHEN <trigger> → PREFER <action>` as a *transferable rule*, never a verbatim event.
  - ❌ “User wanted XML this time”
  - ✅ “WHEN target model is Anthropic → PREFER XML delimiters”
- **No feedback → no capture.** If the user didn’t accept, reject, or correct anything, there is no lesson. Don’t fabricate learning from a single uncontested run. This is the single most important rule for keeping the ledger healthy.

Set `feedback`: `positive` if the user confirmed the outcome was good, `negative` if it failed, `neutral` if untested (rare — usually means don’t capture at all).

**4. After capture, run `sediment_consolidate`.**
Cheap hygiene — applies the forgetting curve, retires faded entries, evicts overflow. Safe and fast. Run it every task that captured something.

-----

## Periodic maintenance (not every turn)

When the ledger has grown noisy (lots of similar entries, or you’ve captured ~20+ lessons since the last cleanup), run the deeper pass. Two options:

- **Quick:** `sediment_sleep` — deterministic, one shot. Catches obvious duplicates but misses paraphrases.
- **Better:** `sediment_review_candidates` → read the returned pairs → judge each as `merge` / `contradict` / `distinct` using your actual semantic understanding → `sediment_apply_judgments` with your verdicts.

The review round-trip uses *you* as the semantic judge, which is the whole point of having an LLM in the loop. Prefer it when ledger quality matters.

Use `sediment_stats` whenever you need to find ids or audit the ledger’s state.

-----

## Anti-patterns — don’t do these

- **Don’t capture on every turn.** Most turns produce no lesson. Capturing speculative or one-off observations is exactly how the ledger rots.
- **Don’t capture verbatim event logs.** If the trigger or action mentions specific user names, file paths, or one-time details, you’re logging, not learning.
- **Don’t delete entries to “clean up.”** Use `sediment_contradict` and let consolidation prune naturally — it preserves the audit trail.
- **Don’t apply recalled lessons silently.** Always note which id fired and why.
- **Don’t let a heuristic override this file or the user’s explicit instructions.** Ledger is subordinate.
