"""
audit.py — observability for Sediment.

Tracks what the ledger actually does in production:
  - Which heuristics fire (and for what queries)
  - Which fired heuristics get confirmed vs contradicted (the "did it help?" signal)
  - Which entries are dead weight (never recalled) vs heavy hitters
  - High-conflict zones — entries that get contradicted a lot

The events are appended to a JSONL file (`events.jsonl`) alongside the ledger.
Append-only is fine here because events ARE the audit trail — they don't get
consolidated. We do trim old events when the file gets huge.

The HTML report is self-contained (no external CDN, no JS framework) — opens
locally, can be emailed, can live in version control.
"""
from __future__ import annotations
import json, os, time, html
from typing import Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter

NOW = lambda: time.time()
DAY = 86400.0


# --------------------------------------------------------------------------- #
# Event types — append-only audit log                                          #
# --------------------------------------------------------------------------- #
def log_event(events_path: str, kind: str, **payload):
    """Append a single event to the JSONL log. Cheap and crash-safe."""
    os.makedirs(os.path.dirname(events_path) or ".", exist_ok=True)
    event = {"t": NOW(), "kind": kind, **payload}
    with open(events_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def read_events(events_path: str, since_days: Optional[float] = None) -> list[dict]:
    if not os.path.exists(events_path):
        return []
    cutoff = (NOW() - since_days * DAY) if since_days else 0
    out = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if e.get("t", 0) >= cutoff:
                    out.append(e)
            except json.JSONDecodeError:
                continue
    return out


def trim_events(events_path: str, keep_days: float = 90.0):
    """Drop events older than keep_days. Run occasionally."""
    if not os.path.exists(events_path):
        return 0
    kept = read_events(events_path, since_days=keep_days)
    with open(events_path, "w") as f:
        for e in kept:
            f.write(json.dumps(e) + "\n")
    return len(kept)


# --------------------------------------------------------------------------- #
# Analysis — turn raw events + ledger into actionable insight                  #
# --------------------------------------------------------------------------- #
@dataclass
class HeuristicReport:
    id: str
    when: str
    prefer: str
    status: str
    confidence: float
    fire_count: int = 0
    confirm_count: int = 0
    contradict_count: int = 0
    last_fired_days_ago: Optional[float] = None
    confirm_rate: float = 0.0          # confirms / (confirms + contradictions)
    health: str = "unknown"            # healthy | drifting | stale | new | dead


def analyze(ledger, events_path: str, since_days: float = 30.0) -> dict:
    """The core insight pass — what's the ledger been doing?"""
    events = read_events(events_path, since_days=since_days)
    now = NOW()

    fires_by_id: Counter = Counter()
    confirms_by_id: Counter = Counter()
    contradicts_by_id: Counter = Counter()
    last_fire_by_id: dict[str, float] = {}
    recall_queries: list[str] = []
    captures_total = 0

    for e in events:
        k = e.get("kind")
        if k == "fire":
            for hid in e.get("ids", []):
                fires_by_id[hid] += 1
                last_fire_by_id[hid] = max(last_fire_by_id.get(hid, 0), e["t"])
            if e.get("query"):
                recall_queries.append(e["query"])
        elif k == "confirm":
            confirms_by_id[e["id"]] += 1
        elif k == "contradict":
            contradicts_by_id[e["id"]] += 1
        elif k == "capture":
            captures_total += 1

    reports: list[HeuristicReport] = []
    for h in ledger.entries:
        fc, cc, xc = fires_by_id[h.id], confirms_by_id[h.id], contradicts_by_id[h.id]
        rate = cc / (cc + xc) if (cc + xc) else 0.0
        last = last_fire_by_id.get(h.id)
        days_ago = (now - last) / DAY if last else None

        # health classification — the actionable judgment
        if h.status != "active":
            health = "retired"
        elif fc == 0 and (now - h.created) / DAY < 7:
            health = "new"
        elif fc == 0:
            health = "dead"          # has existed for a while, never fires — likely useless
        elif xc > cc and (cc + xc) >= 3:
            health = "drifting"      # contradicted more than confirmed — danger
        elif days_ago and days_ago > 30 and h.confirm_count > 0:
            health = "stale"
        else:
            health = "healthy"

        reports.append(HeuristicReport(
            id=h.id, when=h.when, prefer=h.prefer, status=h.status,
            confidence=round(h.effective_confidence(now), 3),
            fire_count=fc, confirm_count=cc, contradict_count=xc,
            last_fired_days_ago=round(days_ago, 1) if days_ago else None,
            confirm_rate=round(rate, 3), health=health,
        ))

    by_health = Counter(r.health for r in reports)
    return {
        "window_days": since_days,
        "total_events": len(events),
        "total_captures": captures_total,
        "total_recall_queries": len(recall_queries),
        "ledger_size_active": len(ledger.active),
        "ledger_size_total": len(ledger.entries),
        "by_health": dict(by_health),
        "drifting": [asdict(r) for r in reports if r.health == "drifting"],
        "dead": [asdict(r) for r in reports if r.health == "dead"],
        "stale": [asdict(r) for r in reports if r.health == "stale"],
        "top_fire": [asdict(r) for r in sorted(reports, key=lambda x: -x.fire_count)[:10] if r.fire_count > 0],
        "all_reports": [asdict(r) for r in sorted(reports, key=lambda x: -x.fire_count)],
        "recent_queries": recall_queries[-20:],
    }


# --------------------------------------------------------------------------- #
# HTML report — self-contained, no external assets                             #
# --------------------------------------------------------------------------- #
_HEALTH_COLORS = {
    "healthy": "#2ea043", "drifting": "#d1242f", "stale": "#bf8700",
    "dead": "#6e7681", "new": "#0969da", "retired": "#8b949e",
}

def render_html(analysis: dict, title: str = "Sediment Ledger Report") -> str:
    h = html.escape
    def chip(text, color):
        return (f'<span style="background:{color};color:#fff;padding:2px 8px;'
                f'border-radius:10px;font-size:11px;font-weight:600">{h(text)}</span>')

    def health_chip(s): return chip(s, _HEALTH_COLORS.get(s, "#6e7681"))

    def row(r: dict) -> str:
        days = f"{r['last_fired_days_ago']}d ago" if r['last_fired_days_ago'] else "—"
        confirm_rate_display = f"{int(r['confirm_rate']*100)}%" if (r['confirm_count'] + r['contradict_count']) else "—"
        return f"""
          <tr>
            <td><code style="font-size:11px;color:#6e7681">{h(r['id'])}</code></td>
            <td>{health_chip(r['health'])}</td>
            <td><b>WHEN</b> {h(r['when'])}<br><span style="color:#57606a">PREFER {h(r['prefer'])}</span></td>
            <td style="text-align:right">{r['confidence']:.2f}</td>
            <td style="text-align:right">{r['fire_count']}</td>
            <td style="text-align:right">{r['confirm_count']}/{r['contradict_count']}</td>
            <td style="text-align:right;color:{'#d1242f' if r['contradict_count']>r['confirm_count'] and r['confirm_count']+r['contradict_count']>=3 else '#1f2328'}">{confirm_rate_display}</td>
            <td style="text-align:right">{days}</td>
          </tr>"""

    def section(title, rows, empty_msg):
        if not rows:
            return f'<section><h2>{h(title)}</h2><p style="color:#6e7681">{h(empty_msg)}</p></section>'
        body = "\n".join(row(r) for r in rows)
        return f"""
        <section>
          <h2>{h(title)} <small style="color:#6e7681;font-weight:normal">({len(rows)})</small></h2>
          <table>
            <thead><tr>
              <th>ID</th><th>Health</th><th>Heuristic</th>
              <th>Conf</th><th>Fires</th><th>+/−</th><th>Rate</th><th>Last fire</th>
            </tr></thead>
            <tbody>{body}</tbody>
          </table>
        </section>"""

    a = analysis
    health_summary = " ".join(
        f'{health_chip(k)} <b>{v}</b>'
        for k, v in sorted(a['by_health'].items(), key=lambda kv: -kv[1])
    )
    queries_html = (
        "<ul style=\"margin:0;padding-left:18px;color:#57606a;font-size:13px\">"
        + "".join(f"<li>{h(q)}</li>" for q in a["recent_queries"][-10:])
        + "</ul>"
    ) if a["recent_queries"] else '<p style="color:#6e7681">No recall queries in window.</p>'

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{h(title)}</title>
<style>
  body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; color: #1f2328;
          max-width: 1100px; margin: 30px auto; padding: 0 20px; background: #fff; }}
  h1 {{ margin: 0 0 4px; font-size: 26px; }}
  h2 {{ margin: 32px 0 12px; font-size: 18px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }}
  .meta {{ color: #6e7681; margin-bottom: 24px; font-size: 13px; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 16px 0 28px; }}
  .stat {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px;
           padding: 12px 16px; min-width: 120px; }}
  .stat .n {{ font-size: 22px; font-weight: 600; display: block; }}
  .stat .l {{ font-size: 11px; color: #6e7681; text-transform: uppercase; letter-spacing: 0.5px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #eaeef2; text-align: left; vertical-align: top; }}
  th {{ background: #f6f8fa; font-weight: 600; color: #57606a; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
  code {{ background: #f6f8fa; padding: 1px 5px; border-radius: 3px; }}
  .legend {{ background: #f6f8fa; border-radius: 8px; padding: 12px 16px; font-size: 12px; color: #57606a; }}
</style></head><body>
<h1>{h(title)}</h1>
<div class="meta">
  Generated {time.strftime('%Y-%m-%d %H:%M')} ·
  Window: last {int(a['window_days'])} days ·
  {a['ledger_size_active']} active / {a['ledger_size_total']} total entries
</div>

<div class="stats">
  <div class="stat"><span class="n">{a['total_recall_queries']}</span><span class="l">Recall queries</span></div>
  <div class="stat"><span class="n">{a['total_captures']}</span><span class="l">New captures</span></div>
  <div class="stat"><span class="n">{a['total_events']}</span><span class="l">Total events</span></div>
</div>

<section>
  <h2>Health summary</h2>
  <p>{health_summary}</p>
  <div class="legend">
    <b>healthy</b>: firing and being confirmed.
    <b>drifting</b>: contradicted more than confirmed — likely wrong now, review or prune.
    <b>stale</b>: hasn't fired in 30+ days but was once confirmed.
    <b>dead</b>: never fires — likely too specific or poorly worded.
    <b>new</b>: created recently, not enough data yet.
  </div>
</section>

{section("⚠️ Drifting — contradicted more than confirmed", a["drifting"],
         "No drifting heuristics. Memory is behaving.")}

{section("💀 Dead weight — never fires", a["dead"],
         "Every heuristic is being recalled. Clean ledger.")}

{section("🕸️ Stale — confirmed once, then forgotten", a["stale"],
         "No stale entries.")}

{section("🔥 Top performers — most-fired heuristics", a["top_fire"],
         "No firing activity in the window.")}

<section>
  <h2>Recent recall queries</h2>
  {queries_html}
</section>

{section("Full ledger", a["all_reports"], "Ledger is empty.")}

</body></html>"""


def write_html_report(ledger, events_path: str, output_path: str,
                      since_days: float = 30.0) -> str:
    a = analyze(ledger, events_path, since_days=since_days)
    html_str = render_html(a)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_str)
    return output_path
