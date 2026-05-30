# Sediment

**Memory consolidation for self-improving AI agents — modeled on how the brain actually does it.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The problem

Naive agent memory is append-only. Every "lesson" gets logged, nothing ever leaves. Within days the ledger fills with:

- **Near-duplicates** — the same lesson captured three times in different words
- **Contradictions** — `"WHEN X → do A"` sitting next to `"WHEN X → do not-A"`, both confidently held
- **Stale entries** — heuristics that were true once, six months ago, in a different context

The ledger eventually **degrades** output instead of improving it. The agent gets dumber the more it "learns."

## The fix

The brain doesn't append. It runs two processes at two different rates:

| Process | Analogue | Cost | Frequency |
|---------|----------|------|-----------|
| **Synaptic homeostasis** | constant decay of weak traces, strengthening of confirmed ones | cheap | every run |
| **Systems consolidation** | sleep — merge, abstract, resolve conflicts | expensive | periodic |

Sediment mirrors both. Memory **shrinks as well as grows**. Forgetting is a feature.

---

## Quick start

```bash
git clone https://github.com/yourname/sediment
cd sediment
pip install -r requirements.txt

# Try the demo — 10 rotted entries, watch them consolidate
python cli.py demo
```

You'll see ten messy entries collapse into a clean six (deterministic) or four (with a semantic judge attached). A contradiction gets preserved as a *boundary rule* on the winner, not deleted.

---

## The Sediment Loop

```
Recall → Apply → Capture → Distill → Consolidate
```

| Phase | What happens |
|-------|--------------|
| **Recall** | Load top-K relevant heuristics into working context |
| **Apply** | Let them inform the task (audibly, with audit trail) |
| **Capture** | Distill the outcome into a `WHEN <trigger> → PREFER <action>` rule |
| **Distill** | Generalize — never log verbatim events |
| **Consolidate** | Decay, prune, merge, abstract — the housekeeping |

Three invariants make the difference between memory that learns and memory that rots:

1. **Externalized, never self-mutating** — the ledger persists in a file; the engine never rewrites the agent's prompt
2. **Forgetting is mandatory** — memory must shrink (decay + prune + evict), not only grow
3. **Generalize, don't just delete** — a resolved contradiction becomes a *boundary rule*, preserving the exception

---

## Two ways to use it

### As a Python library

```python
from sediment import Ledger

L = Ledger("ledger.json").load()

# at the start of a task
hits = L.recall("building a prompt for Claude", k=5)
for h in hits:
    print(f"  applying: {h.when} → {h.prefer}")

# ... do the work ...

# at the end, if the outcome was good
L.capture(
    when="target model is Anthropic",
    prefer="use XML delimiters for structure",
    feedback="positive",
)

# every run — cheap
L.homeostasis()

# periodically — deep
L.sleep()

L.save()
```

### As an MCP server (recommended for agents)

The MCP server exposes the engine as live tools any MCP-capable agent (Claude Code, Claude Desktop, Cursor, etc.) can call mid-conversation.

```bash
python server.py                # stdio (local)
python server.py --http         # streamable HTTP (remote)
```

Wire into Claude Code via `.mcp.json`:

```json
{
  "mcpServers": {
    "sediment": {
      "command": "python3",
      "args": ["/absolute/path/to/server.py"],
      "env": { "SEDIMENT_LEDGER_PATH": "/absolute/path/to/ledger.json" }
    }
  }
}
```

Drop the included `CLAUDE.md` in your project root and the agent will follow the loop discipline automatically.

---

## MCP tools

| Tool | Purpose |
|------|---------|
| `sediment_recall` | Load top-K relevant lessons at task start |
| `sediment_capture` | Distill a real outcome into a generalized heuristic |
| `sediment_confirm` | "That lesson worked" — strengthens + extends half-life |
| `sediment_contradict` | "That lesson failed" — halves confidence, audit trail kept |
| `sediment_consolidate` | Cheap hygiene: decay + retire + evict (every run) |
| `sediment_sleep` | Deep pass with built-in lexical judge (periodic) |
| `sediment_review_candidates` | Surface similar pairs for the **agent** to judge |
| `sediment_apply_judgments` | Apply the agent's merge/contradict verdicts |
| `sediment_stats` | Inspect ledger health |

Standing overhead: **~2,000 tokens** for the full manifest. Per-call traffic is small (~50–300 tokens for the routine loop).

---

## The agent-as-judge pattern

The deterministic `sleep` pass works out of the box but can't tell paraphrases apart — `"target model is Claude"` and `"Anthropic model as target"` look like distinct entries to a lexical matcher.

The fix isn't to embed an LLM inside the server. **The calling agent is already an LLM.**

`sediment_review_candidates` returns the pairs that need a semantic call. The agent reads them, decides `merge` / `contradict` / `distinct`, and sends the verdicts to `sediment_apply_judgments`. The server applies them deterministically. Best of both — the agent's intelligence makes the hard call, the server enforces the invariants.

---

## How it actually works under the hood

### Forgetting curve + spaced repetition

Every heuristic has a half-life that starts at ~14 days. Each confirmation **extends** the half-life:

```python
half_life_days = 14.0 * (1.0 + confirm_count)
effective_confidence = stored_confidence * 0.5 ** (days_since_confirmed / half_life)
```

Useful lessons migrate toward permanence. One-off noise fades on its own. Exactly the mechanism behind why you remember things you've used many times and forget things you've used once.

### Consolidation: merge, resolve, abstract

The `sleep` pass does three structural operations:

- **Merge**: same trigger + same action → fold into one entry, sum hit counts
- **Resolve contradiction**: same trigger + opposite action → keep the stronger one, encode the loser as a *boundary exception* in a note (`"usually X; exception toward Y when context differs"`)
- **Abstract**: 3+ siblings with the same action across similar triggers → create a parent heuristic, mark the children dormant

Contradictions becoming boundary rules is the part that actually grows wisdom rather than just pruning. The conflict itself is information about where the rule's edges are.

### Budget cap + eviction

The ledger has a cap (default 200 active entries). When over, the lowest **retention value** entries get evicted:

```python
retention = effective_confidence × (1 + log(hit_count)) × (1 + 0.2 × confirm_count)
```

Frequently-applied, often-confirmed lessons survive. Rarely-touched ones go first.

---

## Project structure

```
sediment/
├── sediment.py          # core engine (zero dependencies)
├── server.py            # MCP server (FastMCP)
├── cli.py               # demo + CLI
├── CLAUDE.md            # drop in your project root for Claude Code
├── requirements.txt
└── README.md
```

`sediment.py` runs anywhere with stdlib only. `server.py` adds the MCP layer.

---

## When to use Sediment

**Good fit:**
- Long-running agents that should improve at recurring task types
- Domain-specific assistants where you'd otherwise hand-tune the prompt over time
- Multi-session workflows where the agent needs continuity beyond context window
- Any case where you've thought "I wish it remembered what worked last time"

**Bad fit:**
- One-shot tasks with no recurring patterns
- Tasks where every run is genuinely unique (no transferable lessons exist)
- Agents that should be fully deterministic (use a static prompt instead)

---

## Design decisions worth knowing

**Why externalized state, not prompt mutation?** A system prompt can't rewrite itself — calls are stateless. "Self-improving" agents that pretend otherwise are usually hollow. Sediment is honest: the prompt is fixed, the *prompt + ledger as a system* is what learns. Externalizing also means the memory survives crashes, model swaps, and process restarts.

**Why is consolidation two passes instead of one?** Mixing them is why most agent-memory implementations either run too slow to use often or too dumb to be worth running. Decay and eviction are O(n) and cheap — they belong on every run. Semantic merging is O(n²) and expensive — it belongs on a schedule.

**Why preserve contradictions as boundary rules instead of deleting losers?** Because the contradiction *is* information. "Usually X, but Y when..." is a richer rule than just "X." Deletion throws away the signal that the rule has edges.

**Why is the judge pluggable?** The deterministic lexical judge is good enough for production, ships with zero dependencies, and never fails opaquely. The LLM judge is better but costs API calls and adds a failure mode. Most users should start lexical and upgrade only if quality matters more than cost.

---

## Roadmap

- [ ] Embedding-based similarity (swap `text_sim` for a real vector model)
- [ ] Auto-scheduled `sleep` based on redundancy pressure, not turn count
- [ ] Multi-agent shared ledgers with conflict resolution
- [ ] SQLite backend for very large ledgers
- [ ] Optional telemetry: which lessons fire most, which get pruned, drift over time

---

## License

MIT