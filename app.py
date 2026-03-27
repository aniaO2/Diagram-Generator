from sklearn.preprocessing import scale
import streamlit as st
import base64
from datetime import datetime
import re

# Importăm logica din backend
from backend import call_ollama, DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL

st.set_page_config(page_title="NL2Diagram", page_icon="⬡", layout="wide", initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════════
# CSS — blueprint / terminal aesthetic
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Fira+Code:wght@300;400;500&display=swap');

:root {
  --bg:       #060a0f;
  --surface:  #0c1219;
  --card:     #101820;
  --border:   #1a2a3a;
  --border2:  #243448;
  --cyan:     #00d4ff;
  --cyan2:    #0099cc;
  --green:    #00ff88;
  --red:      #ff4757;
  --amber:    #ffa502;
  --purple:   #a855f7;
  --text:     #d4e8f0;
  --muted:    #4a6a80;
  --mono:     'Fira Code', monospace;
  --sans:     'Syne', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
}

[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; }
.main .block-container { padding: 1.2rem 1.8rem !important; max-width: 100% !important; position: relative; z-index: 1; }

/* ── Header ── */
.nl2d-header { display: flex; align-items: flex-end; gap: 16px; margin-bottom: 1.2rem; padding-bottom: 12px; border-bottom: 1px solid var(--border2); }
.nl2d-logo { font-family: var(--sans); font-size: 1.7rem; font-weight: 800; color: var(--cyan); letter-spacing: -.02em; line-height: 1; }
.nl2d-logo span { color: var(--text); }
.nl2d-tagline { font-family: var(--mono); font-size: .7rem; color: var(--muted); letter-spacing: .08em; padding-bottom: 3px; }
.nl2d-version { margin-left: auto; font-family: var(--mono); font-size: .65rem; color: var(--muted); border: 1px solid var(--border2); padding: 3px 8px; border-radius: 3px; }

/* ── Panel labels ── */
.panel-label { font-family: var(--mono); font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.panel-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── Chat ── */
.chat-wrap { background: var(--surface); border: 1px solid var(--border2); border-radius: 10px; padding: 12px; min-height: 320px; max-height: 360px; overflow-y: auto; margin-bottom: 10px; }
.chat-wrap::-webkit-scrollbar { width: 3px; }
.chat-wrap::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

.msg-user { display: flex; justify-content: flex-end; margin: 6px 0; }
.msg-user .bubble { background: linear-gradient(135deg, #0d2235, #0c1a28); border: 1px solid var(--border2); border-radius: 12px 12px 3px 12px; padding: 8px 12px; max-width: 85%; font-size: .83rem; color: var(--text); line-height: 1.5; }

.msg-ai { display: flex; gap: 8px; margin: 6px 0; }
.msg-ai .avatar { width: 26px; height: 26px; flex-shrink: 0; background: linear-gradient(135deg, var(--cyan2), #004466); border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: .7rem; color: #fff; font-family: var(--mono); font-weight: 500; }
.msg-ai .bubble { background: var(--card); border: 1px solid var(--border); border-left: 2px solid var(--cyan); border-radius: 3px 12px 12px 12px; padding: 8px 12px; max-width: 88%; font-size: .83rem; color: var(--text); line-height: 1.5; }
.msg-clarify .bubble { border-left-color: var(--amber) !important; background: linear-gradient(135deg, #1a1200, #101820) !important; }
.msg-guard .bubble { border-left-color: var(--red) !important; background: linear-gradient(135deg, #1a0408, #101820) !important; }

.badge { display: inline-flex; align-items: center; gap: 4px; font-family: var(--mono); font-size: .65rem; padding: 2px 7px; border-radius: 3px; margin: 3px 2px 0; }
.badge-fix  { background: rgba(255,165,2,.1); border: 1px solid var(--amber); color: var(--amber); }
.badge-note { background: rgba(168,85,247,.1); border: 1px solid var(--purple); color: var(--purple); }
.badge-guard{ background: rgba(255,71,87,.1);  border: 1px solid var(--red);   color: var(--red); }

.empty-chat { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 220px; color: var(--muted); font-size: .8rem; text-align: center; line-height: 2; font-family: var(--mono); }
.empty-chat .icon { font-size: 2rem; margin-bottom: 8px; opacity: .3; }

/* ── Inputs / Buttons ── */
.stTextArea textarea { background: var(--card) !important; border: 1px solid var(--border2) !important; color: var(--text) !important; border-radius: 8px !important; font-family: var(--mono) !important; font-size: .85rem !important; line-height: 1.6 !important; resize: none !important; }
.stTextArea textarea:focus { border-color: var(--cyan) !important; box-shadow: 0 0 0 1px rgba(0,212,255,.2) !important; }
.stButton > button { background: var(--card) !important; border: 1px solid var(--border2) !important; color: var(--text) !important; border-radius: 7px !important; font-family: var(--sans) !important; font-size: .82rem !important; font-weight: 600 !important; transition: all .18s ease !important; }
.stButton > button:hover { border-color: var(--cyan) !important; color: var(--cyan) !important; background: rgba(0,212,255,.05) !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #00aacc, #0077aa) !important; border-color: var(--cyan) !important; color: #fff !important; }
[data-testid="stDownloadButton"] button { font-size: .75rem !important; padding: 4px 10px !important; color: var(--muted) !important; }

/* ── Canvas / Pills ── */
.canvas-wrap { background: var(--surface); border: 1px solid var(--border2); border-radius: 10px; overflow: hidden; position: relative; }
.canvas-top { display: flex; align-items: center; gap: 8px; padding: 8px 14px; background: var(--card); border-bottom: 1px solid var(--border); font-family: var(--mono); font-size: .7rem; color: var(--muted); }
.canvas-dot { width: 8px; height: 8px; border-radius: 50%; }
.dot-r { background: #ff5f57; } .dot-y { background: #febc2e; } .dot-g { background: #28c840; }

.pill { display: inline-flex; align-items: center; gap: 5px; font-family: var(--mono); font-size: .65rem; padding: 3px 9px; border-radius: 20px; margin: 0 2px; }
.pill-cyan   { background: rgba(0,212,255,.1); border: 1px solid rgba(0,212,255,.4); color: var(--cyan); }
.pill-green  { background: rgba(0,255,136,.1); border: 1px solid rgba(0,255,136,.4); color: var(--green); }
.pill-amber  { background: rgba(255,165,2,.1);  border: 1px solid rgba(255,165,2,.4);  color: var(--amber); }
.pill-purple { background: rgba(168,85,247,.1); border: 1px solid rgba(168,85,247,.4); color: var(--purple); }
.pill-red    { background: rgba(255,71,87,.1);  border: 1px solid rgba(255,71,87,.4);  color: var(--red); }

/* ── Extras ── */
.hist-entry { background: var(--card); border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px; margin: 4px 0; font-size: .75rem; color: var(--muted); cursor: pointer; font-family: var(--mono); overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.hist-entry:hover { border-color: var(--cyan); color: var(--text); }
.streamlit-expanderHeader { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 7px !important; color: var(--text) !important; font-family: var(--mono) !important; font-size: .78rem !important; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 10px 0 !important; }
[data-testid="stSpinner"] { color: var(--cyan) !important; }
.stSelectbox > div > div { background: var(--card) !important; border: 1px solid var(--border2) !important; color: var(--text) !important; border-radius: 7px !important; font-size: .82rem !important; }
[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; padding: 10px 14px !important; }
[data-testid="stMetricLabel"] { font-family: var(--mono) !important; font-size: .65rem !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-family: var(--sans) !important; font-size: 1.1rem !important; color: var(--cyan) !important; }
h3 { font-family: var(--mono) !important; font-size: .68rem !important; color: var(--muted) !important; text-transform: uppercase; letter-spacing: .1em; font-weight: 400 !important; margin-bottom: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MERMAID RENDERER
# ══════════════════════════════════════════════════════════
MERMAID_THEMES = {"default": "default", "dark": "dark", "forest": "forest", "neutral": "neutral", "base": "base"}


def render_mermaid(code: str, theme: str = "default", height: int = 650) -> str:
    bg = "#ffffff" if theme != "dark" else "#1a1a2e"
    text = "#111111" if theme != "dark" else "#e8eef5"

    safe_code = code.replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    background: {bg};
    color: {text};
    font-family: Arial, sans-serif;
    overflow: hidden;
  }}

  #outer {{
    width: 100vw;
    height: {height}px;
    position: relative;
    background: {bg};
    overflow: auto;
  }}

  #container {{
    min-width: 100%;
    min-height: 100%;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 24px;
  }}

  #diagram {{
    transform-origin: top center;
    transition: transform 0.15s ease;
    cursor: grab;
    user-select: none;
  }}

  #diagram:active {{
    cursor: grabbing;
  }}

  .mermaid {{
    display: inline-block;
  }}

  svg {{
    width: auto !important;
    height: auto !important;
    max-width: none !important;
    min-width: 300px;
  }}

  #controls {{
    position: sticky;
    bottom: 12px;
    float: right;
    display: flex;
    gap: 6px;
    z-index: 20;
    margin: 0 12px 12px 0;
  }}

  #controls button {{
    background: rgba(0,0,0,.7);
    color: #fff;
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    cursor: pointer;
  }}

  #controls button:hover {{
    background: rgba(0,150,200,.75);
  }}

  .error {{
    color: #ff4757;
    font-family: monospace;
    white-space: pre-wrap;
    padding: 20px;
  }}
</style>
</head>
<body>
  <div id="outer">
    <div id="container">
      <div id="diagram">
        <pre class="mermaid">{safe_code}</pre>
      </div>
    </div>
    <div id="controls">
      <button onclick="zoom(1.15)">＋</button>
      <button onclick="zoom(0.87)">－</button>
      <button onclick="resetView()">⊙</button>
    </div>
  </div>

  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

    mermaid.initialize({{
      startOnLoad: true,
      theme: '{theme}',
      securityLevel: 'loose',
      fontFamily: 'Arial, sans-serif',
      themeVariables: {{
        fontSize: '18px',
        primaryTextColor: '{text}',
        lineColor: '{text}',
        secondaryColor: '#dde7f0',
        tertiaryColor: '#f4f7fb'
      }},
      flowchart: {{
        useMaxWidth: false,
        htmlLabels: true,
        curve: 'linear',
        nodeSpacing: 50,
        rankSpacing: 70,
        padding: 20
      }},
      sequence: {{
        useMaxWidth: false,
        wrap: true,
        actorMargin: 80,
        width: 180,
        height: 70,
        boxMargin: 12,
        boxTextMargin: 8,
        noteMargin: 12,
        messageMargin: 45
      }},
      er: {{
        useMaxWidth: false
      }},
      gantt: {{
        useMaxWidth: false
      }}
    }});

    let scale = 1;
    let tx = 0;
    let ty = 0;
    const el = document.getElementById('diagram');

    function apply() {{
      el.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
    }}

    window.zoom = function(factor) {{
      scale = Math.min(2.5, Math.max(0.5, scale * factor));
      apply();
    }}

    window.resetView = function() {{
      scale = 1;
      tx = 0;
      ty = 0;
      apply();
    }}

    let dragging = false;
    let startX = 0;
    let startY = 0;

    el.addEventListener('mousedown', (e) => {{
      dragging = true;
      startX = e.clientX - tx;
      startY = e.clientY - ty;
    }});

    document.addEventListener('mouseup', () => {{
      dragging = false;
    }});

    document.addEventListener('mousemove', (e) => {{
      if (!dragging) return;
      tx = e.clientX - startX;
      ty = e.clientY - startY;
      apply();
    }});

    el.addEventListener('wheel', (e) => {{
      e.preventDefault();
      zoom(e.deltaY < 0 ? 1.08 : 0.92);
    }}, {{ passive: false }});

    window.addEventListener('error', (e) => {{
      document.body.innerHTML = `<div class="error">Mermaid render error:\\n${{e.message}}</div>`;
    }});
  </script>
</body>
</html>"""

def sanitize_mermaid(code: str) -> str:
    if not code:
        return "flowchart TD\n    A[Empty diagram]"
    code = code.strip()
    code = re.sub(r"^```mermaid\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"^```\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    lines = [line.rstrip() for line in code.splitlines()]
    return "\n".join(lines).strip()

def looks_like_mermaid(code: str) -> bool:
    starters = (
        "flowchart", "graph", "sequenceDiagram", "classDiagram",
        "erDiagram", "stateDiagram", "stateDiagram-v2", "gantt"
    )
    stripped = code.strip()
    return stripped.startswith(starters)

def estimate_node_count(mermaid_code: str) -> int:
    patterns = [
        r'\b[A-Za-z0-9_]+\[',     
        r'\b[A-Za-z0-9_]+\(',     
        r'\b[A-Za-z0-9_]+\{',     
        r'\b[A-Za-z0-9_]+\[\(',   
    ]
    count = 0
    for p in patterns:
        count += len(re.findall(p, mermaid_code))
    return max(count, 1)

# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
defaults = {
    "conv_history": [],
    "diagrams": [],
    "current_diag": None,
    "total_generated": 0,
    "theme": "default",
    "ollama_model": DEFAULT_OLLAMA_MODEL
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="nl2d-header">
  <div><div class="nl2d-logo">NL2<span>DIAGRAM</span></div><div class="nl2d-tagline">// natural language → architecture diagrams</div></div>
  <div class="nl2d-version">QuantChallenge 2026 · v2.0</div>
</div>
""", unsafe_allow_html=True)

col_left, col_chat, col_canvas = st.columns([0.7, 1.1, 1.9], gap="medium")

# ── LEFT PANEL ──────────────────────────────
with col_left:
    st.markdown("### Stats")
    st.metric("Generated", st.session_state.total_generated)
    st.metric("In History", len(st.session_state.diagrams))

    if st.session_state.current_diag:
        d = st.session_state.current_diag
        st.metric("Confidence", f"{d.get('confidence', 0)}%")
        st.metric("Nodes", d.get("node_count", "—"))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Theme")
    theme = st.selectbox("Diagram theme", list(MERMAID_THEMES.keys()),
                         index=list(MERMAID_THEMES.keys()).index(st.session_state.theme), label_visibility="collapsed")
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Model")
    new_model = st.text_input("Ollama model", value=st.session_state.ollama_model, label_visibility="collapsed")
    if new_model != st.session_state.ollama_model:
        st.session_state.ollama_model = new_model

    if st.session_state.diagrams:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### History")
        for i, d in enumerate(reversed(st.session_state.diagrams[-8:])):
            dtype_icon = {"flowchart": "⬡", "sequenceDiagram": "⇄", "erDiagram": "⊞", "classDiagram": "◈",
                          "stateDiagram-v2": "◎", "gantt": "▦"}.get(d.get("dtype", ""), "⬡")
            if st.button(f"{dtype_icon} {d['label']}", key=f"hist_{i}", use_container_width=True):
                st.session_state.current_diag = d
                st.rerun()

# ── CHAT PANEL ──────────────────────────────
with col_chat:
    st.markdown("### Chat")
    chat_html = '<div class="chat-wrap" id="chat">'
    if not st.session_state.conv_history:
        chat_html += """<div class="empty-chat"><div class="icon">⬡</div>describe any diagram<br>typos are fine · follow-ups work<br><span style="opacity:.5">——————————</span><br>orchestrator pipelines<br>database schemas<br>state machines · sequence flows</div>"""
    for msg in st.session_state.conv_history:
        if msg["role"] == "user":
            chat_html += f'<div class="msg-user"><div class="bubble">{msg["display"]}</div></div>'
        else:
            extra_class = "msg-guard" if msg.get("is_guard") else "msg-clarify" if msg.get("is_clarify") else ""
            chat_html += f'<div class="msg-ai {extra_class}"><div class="avatar">AI</div><div class="bubble">{msg["display"]}'
            for c in msg.get("corrections", []): chat_html += f'<br><span class="badge badge-fix">✏ {c}</span>'
            for a in msg.get("assumptions", []): chat_html += f'<br><span class="badge badge-note">◆ {a}</span>'
            if msg.get("is_guard"): chat_html += f'<br><span class="badge badge-guard">⚠ guardrail triggered</span>'
            chat_html += "</div></div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    user_input = st.text_area("query", height=100, key="input_area", label_visibility="collapsed",
                              placeholder="Describe your diagram…\ne.g. 'orchestrator with azure guardrails routes...'")

    c1, c2, c3 = st.columns([5, 1, 1])
    with c1:
        go = st.button("⬡ Generate", type="primary", use_container_width=True)
    with c2:
        regen = st.button("↺", use_container_width=True, help="Regenerate last")
    with c3:
        clr = st.button("✕", use_container_width=True, help="Clear all")

    if clr:
        for k in ["conv_history", "diagrams", "current_diag"]:
            st.session_state[k] = [] if k != "current_diag" else None
        st.rerun()

    trigger_input = None
    if go and user_input.strip():
        trigger_input = user_input.strip()
    elif regen and st.session_state.conv_history:
        last_user = next((m["raw"] for m in reversed(st.session_state.conv_history) if m["role"] == "user"), None)
        if last_user:
            trigger_input = last_user
            st.session_state.conv_history = [m for m in st.session_state.conv_history if not (m["role"] == "assistant")]

    if trigger_input:
        with st.spinner("Generating diagram…"):
            parsed, raw = call_ollama(
                st.session_state.conv_history,
                trigger_input,
                model=st.session_state.ollama_model
            )

        st.session_state.conv_history.append({"role": "user", "display": trigger_input, "raw": trigger_input})

        is_guard = parsed.get("guardrail_triggered", False)
        is_clarify = parsed.get("needs_clarification", False) and not is_guard

        if is_guard:
            display_msg = f"⚠ Request blocked: {parsed.get('guardrail_reason', 'Policy violation detected.')}"
        elif is_clarify:
            display_msg = parsed.get("clarification_question", "Could you clarify your request?")
        else:
            display_msg = parsed.get("user_message", "Diagram generated.")

        st.session_state.conv_history.append({
            "role": "assistant", "display": display_msg or "Done.", "raw": raw,
            "is_guard": is_guard, "is_clarify": is_clarify,
            "corrections": parsed.get("corrections_made", []), "assumptions": parsed.get("assumptions_made", [])
        })

        if not is_guard and not is_clarify and parsed.get("mermaid_code"):
            clean_code = sanitize_mermaid(parsed["mermaid_code"])

            if not looks_like_mermaid(clean_code):
                clean_code = "flowchart TD\n    A[Invalid Mermaid output]\n    B[Try regenerate or edit manually]\n    A --> B"
            label = trigger_input[:40] + ("…" if len(trigger_input) > 40 else "")
            entry = {
                "mermaid_code": clean_code, "description": parsed.get("diagram_description", ""),
                "label": label, "ts": datetime.now().strftime("%H:%M:%S"),
                "dtype": parsed.get("diagram_type", "flowchart"), "confidence": parsed.get("confidence", 75),
                "node_count": estimate_node_count(clean_code), "corrections": parsed.get("corrections_made", []),
                "assumptions": parsed.get("assumptions_made", []),
            }
            st.session_state.diagrams.append(entry)
            st.session_state.current_diag = entry
            st.session_state.total_generated += 1

        st.rerun()

# ── CANVAS PANEL ──────────────────────────────
with col_canvas:
    st.markdown("### Canvas")
    if st.session_state.current_diag:
        d = st.session_state.current_diag
        conf = d.get("confidence", 0)
        conf_color = "green" if conf >= 80 else "amber" if conf >= 60 else "red"

        pills = f'<span class="pill pill-cyan">⬡ {d.get("dtype", "flowchart")}</span><span class="pill pill-{conf_color}">◈ {conf}% confidence</span><span class="pill pill-purple">⊞ ~{d.get("node_count", "?")} nodes</span>'
        if d.get("corrections"): pills += f'<span class="pill pill-amber">✏ {len(d["corrections"])} fixes</span>'

        st.markdown(f"""
        <div class="canvas-wrap">
          <div class="canvas-top">
            <span class="canvas-dot dot-r"></span><span class="canvas-dot dot-y"></span><span class="canvas-dot dot-g"></span>
            <span style="margin-left:6px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{d.get("description", "diagram")}</span>
            <span style="margin-left:auto">{d['ts']}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        diagram_type = d.get("dtype", "flowchart")

        dynamic_height = {
            "sequenceDiagram": 760,
            "flowchart": 680,
            "erDiagram": 700,
            "classDiagram": 700,
            "stateDiagram-v2": 680,
            "gantt": 720
        }.get(diagram_type, 680)

        st.components.v1.html(
            render_mermaid(
                d["mermaid_code"],
                theme=MERMAID_THEMES[st.session_state.theme],
                height=dynamic_height
            ),
            height=dynamic_height,
            scrolling=True
        )

        with st.expander("⬡ Edit · Export"):
            edited = st.text_area("mermaid", value=d["mermaid_code"], height=200, label_visibility="collapsed",
                                  key="editor")
            cols = st.columns(4)
            with cols[0]:
                if st.button("Apply", type="primary", use_container_width=True):
                    st.session_state.current_diag["mermaid_code"] = edited
                    for item in st.session_state.diagrams:
                        if item is st.session_state.current_diag: item["mermaid_code"] = edited
                    st.rerun()
            with cols[1]:
                st.download_button("⬇ .mmd", data=d["mermaid_code"], file_name="diagram.mmd", mime="text/plain",
                                   use_container_width=True)
            with cols[2]:
                html_export = render_mermaid(d["mermaid_code"], theme=MERMAID_THEMES[st.session_state.theme])
                st.download_button("⬇ .html", data=html_export, file_name="diagram.html", mime="text/html",
                                   use_container_width=True)
            with cols[3]:
                b64 = base64.b64encode(d["mermaid_code"].encode()).decode()
                st.markdown(
                    f"""<button onclick="navigator.clipboard.writeText(atob('{b64}')).then(()=>this.textContent='✓ Copied').catch(()=>this.textContent='✗ Error');setTimeout(()=>this.textContent='Copy',2000)" style="width:100%;background:#101820;border:1px solid #1a2a3a;color:#4a6a80;border-radius:7px;padding:6px 0;font-size:.78rem;cursor:pointer;font-family:'Fira Code',monospace;transition:all .2s">Copy</button>""",
                    unsafe_allow_html=True)

        if d.get("corrections") or d.get("assumptions"):
            with st.expander("✏ Corrections & Assumptions"):
                if d.get("corrections"):
                    st.markdown("**Typo corrections:**")
                    for c in d["corrections"]: st.markdown(f'<span class="badge badge-fix">✏ {c}</span>',
                                                           unsafe_allow_html=True)
                if d.get("assumptions"):
                    st.markdown("**Assumptions made:**")
                    for a in d["assumptions"]: st.markdown(f'<span class="badge badge-note">◆ {a}</span>',
                                                           unsafe_allow_html=True)

    else:
        st.components.v1.html(
            """<style>body{margin:0;background:#060a0f;} .grid{position:fixed;inset:0;background-image:linear-gradient(rgba(0,212,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,.03) 1px,transparent 1px);background-size:40px 40px;} .wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:500px;font-family:'Fira Code',monospace;position:relative;z-index:1;} .hex{font-size:3.5rem;opacity:.15;line-height:1;margin-bottom:20px;color:#00d4ff;} .title{font-size:.95rem;color:#2a4a60;font-weight:500;margin-bottom:12px;} .list{font-size:.72rem;color:#1a2a3a;line-height:2.2;text-align:center;} .border{border:1px dashed #1a2a3a;border-radius:10px;padding:40px 60px;}</style><div class="grid"></div><div class="wrap"><div class="border"><div class="hex">⬡</div><div class="title">// awaiting diagram</div><div class="list">flowchart · sequenceDiagram<br>erDiagram · classDiagram<br>stateDiagram · gantt<br>——<br>zoom + pan · live edit<br>export .mmd / .html<br>typo resilience · guardrails</div></div></div>""",
            height=520)