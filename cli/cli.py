"""Demo + CLI for the Sediment engine.  Run:  python cli.py [demo|llm]"""
import sys, time
from sediment import Ledger, Heuristic, show, DAY

def seed_rotting_ledger() -> Ledger:
    """A ledger exhibiting all three rot failure modes."""
    L = Ledger(path="demo.json", cap=50)
    now = time.time()
    raw = [
        # REDUNDANCY: three near-duplicate lessons phrased differently
        ("target model is Claude", "use XML delimiter tags for structure", 4, 0.6),
        ("when the model is Claude / Anthropic", "wrap sections in XML tags", 2, 0.5),
        ("Anthropic model as target", "structure with XML style tags", 3, 0.55),
        # CONTRADICTION: same trigger, opposite actions
        ("non-reasoning GPT target", "inject explicit step-by-step CoT", 5, 0.7),
        ("non-reasoning GPT target", "do not inject CoT, keep it terse", 1, 0.3),
        # STALE: confirmed once long ago, never fired since
        ("user asks for a haiku generator", "default to 5-7-5 syllables", 1, 0.5),
        # GENERALIZABLE cluster: specific siblings of one parent rule
        ("trading prompt for XAUUSD", "anchor reasoning on HTF bias first", 3, 0.6),
        ("trading prompt for crypto perps", "anchor reasoning on HTF bias first", 2, 0.55),
        ("trading prompt for futures", "anchor reasoning on HTF bias first", 2, 0.55),
        # a healthy unique lesson
        ("output must be valid JSON", "append a self-verification checklist step", 6, 0.8),
    ]
    for when, prefer, conf_n, base in raw:
        L.entries.append(Heuristic(when=when, prefer=prefer, confidence=base,
                                   confirm_count=conf_n, hit_count=conf_n))
    L.entries[5].last_confirmed = now - 120 * DAY   # make haiku lesson stale
    return L

def demo():
    L = seed_rotting_ledger()
    show(L, "BEFORE — the rotted ledger")
    print("\n>>> PASS 1: synaptic homeostasis (decay + retire stale + budget)")
    print("   ", L.homeostasis())
    print("\n>>> PASS 2: sleep — merge / resolve contradictions / abstract (lexical fallback)")
    print("   ", L.sleep())
    show(L, "AFTER — consolidated (deterministic only)")

def demo_llm():
    L = seed_rotting_ledger()
    L.homeostasis()
    def llm_judge(a, b):
        xml = lambda h: any(w in (h.when + h.prefer).lower()
                            for w in ("claude", "anthropic", "xml"))
        return "merge" if xml(a) and xml(b) else L._lexical_judge(a, b)
    print(">>> PASS 2 with a (mock) LLM judge attached")
    print("   ", L.sleep(judge=llm_judge))
    show(L, "AFTER — consolidated WITH semantic judge")

CMDS = {"demo": demo, "llm": demo_llm}
if __name__ == "__main__":
    CMDS.get(sys.argv[1] if len(sys.argv) > 1 else "demo", demo)()
