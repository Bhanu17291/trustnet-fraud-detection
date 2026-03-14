"""
TrustNet — Step 7: Fraud Network Graph
Visualizes connected fraud rings using NetworkX + PyVis
Place this file in: src/step7_network_graph.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import pickle
import os
import tempfile

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DATA_PATH   = os.path.join(MODELS_DIR, 'processed_data.pkl')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrustNet — Fraud Network",
    page_icon="🕸️",
    layout="wide"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #0a0e1a;
        color: #e0e6f0;
    }
    .main { background-color: #0a0e1a; }

    .header-container {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a0a2e 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .header-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,255,200,0.02) 2px,
            rgba(0,255,200,0.02) 4px
        );
        pointer-events: none;
    }
    .header-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.2rem;
        color: #00ffc8;
        margin: 0;
        text-shadow: 0 0 20px rgba(0,255,200,0.4);
        letter-spacing: 2px;
    }
    .header-sub {
        color: #7090b0;
        font-size: 1rem;
        margin-top: 0.3rem;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
    }

    .stat-card {
        background: linear-gradient(135deg, #0d1b2a, #111827);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.3s;
    }
    .stat-card:hover { border-color: #00ffc8; }
    .stat-number {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2rem;
        font-weight: bold;
        color: #00ffc8;
        text-shadow: 0 0 10px rgba(0,255,200,0.3);
    }
    .stat-label {
        font-size: 0.85rem;
        color: #7090b0;
        margin-top: 0.2rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .ring-card {
        background: linear-gradient(135deg, #1a0a0a, #0d1b2a);
        border: 1px solid #ff4444;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: all 0.2s;
    }
    .ring-card:hover {
        border-color: #ff8888;
        transform: translateX(4px);
    }
    .ring-id {
        font-family: 'Share Tech Mono', monospace;
        color: #ff6666;
        font-size: 0.9rem;
    }
    .ring-stats { color: #a0b4c8; font-size: 0.85rem; margin-top: 0.3rem; }

    .section-title {
        font-family: 'Share Tech Mono', monospace;
        color: #00ffc8;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        color: #7090b0 !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.85rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #003d2e, #001a3a) !important;
        color: #00ffc8 !important;
        border: 1px solid #00ffc8 !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 2px !important;
        border-radius: 6px !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stButton"] button:hover {
        background: #00ffc8 !important;
        color: #0a0e1a !important;
    }

    .alert-box {
        background: linear-gradient(135deg, #1a0505, #0d0d1a);
        border: 1px solid #ff4444;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-family: 'Share Tech Mono', monospace;
        color: #ff8888;
        font-size: 0.9rem;
    }
    .info-box {
        background: linear-gradient(135deg, #001a1a, #001a3a);
        border: 1px solid #00ffc8;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-family: 'Share Tech Mono', monospace;
        color: #00ffc8;
        font-size: 0.85rem;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        X_test  = data.get('X_test', data.get('X_train'))
        y_test  = data.get('y_test', data.get('y_train'))
    else:
        X_test, y_test = data
    df = pd.DataFrame(X_test)
    df.columns = [f'V{i}' if i > 0 else 'Time'
                  for i in range(len(df.columns))]
    if 'Amount' not in df.columns:
        df.columns = list(df.columns[:-1]) + ['Amount']
    df['Class'] = np.array(y_test)
    return df


# ── Graph builder ──────────────────────────────────────────────────────────────
def build_fraud_graph(df, n_fraud=80, time_window=3600, amount_tolerance=0.15):
    """
    Build a fraud network graph.
    Edges connect transactions that share:
      - Similar time window (same session / velocity burst)
      - Similar amount (round-tripping / structuring)
    Nodes are transactions; colour = fraud/legit.
    """
    fraud_df = df[df['Class'] == 1].head(n_fraud).copy().reset_index(drop=True)
    legit_df = df[df['Class'] == 0].sample(
        min(n_fraud // 3, len(df[df['Class'] == 0])),
        random_state=42
    ).copy().reset_index(drop=True)

    combined = pd.concat([fraud_df, legit_df], ignore_index=True)
    combined['node_id'] = combined.index
    combined['label']   = combined['Class'].map({1: 'FRAUD', 0: 'LEGIT'})

    G = nx.Graph()

    for _, row in combined.iterrows():
        nid = int(row['node_id'])
        amt = float(row['Amount']) if 'Amount' in row else 0.0
        G.add_node(
            nid,
            label   = f"TX-{nid:04d}",
            title   = (f"<b>TX-{nid:04d}</b><br>"
                       f"Type: {row['label']}<br>"
                       f"Amount: ${amt:.2f}<br>"
                       f"Time: {float(row.get('Time', 0)):.0f}s"),
            color   = '#ff4444' if row['Class'] == 1 else '#1e5f8a',
            size    = 18 if row['Class'] == 1 else 10,
            group   = int(row['Class']),
        )

    # Connect nodes with similar time + amount (fraud ring signature)
    times   = combined['Time'].values   if 'Time'   in combined.columns else np.zeros(len(combined))
    amounts = combined['Amount'].values if 'Amount' in combined.columns else np.zeros(len(combined))

    for i in range(len(combined)):
        for j in range(i + 1, len(combined)):
            dt = abs(float(times[i])   - float(times[j]))
            da = abs(float(amounts[i]) - float(amounts[j]))
            avg_amt = (abs(float(amounts[i])) + abs(float(amounts[j]))) / 2 + 1e-6
            if dt < time_window and da / avg_amt < amount_tolerance:
                both_fraud = combined.iloc[i]['Class'] == 1 and combined.iloc[j]['Class'] == 1
                G.add_edge(
                    int(combined.iloc[i]['node_id']),
                    int(combined.iloc[j]['node_id']),
                    color  = '#ff6666' if both_fraud else '#2a4a6a',
                    width  = 3 if both_fraud else 1,
                    title  = 'Fraud-Fraud link' if both_fraud else 'Mixed link',
                )

    return G, combined


def detect_fraud_rings(G, combined):
    """Find connected components that are mostly fraud."""
    rings = []
    for component in nx.connected_components(G):
        if len(component) < 2:
            continue
        nodes     = list(component)
        fraud_cnt = sum(1 for n in nodes if G.nodes[n].get('group') == 1)
        if fraud_cnt >= 2:
            rings.append({
                'nodes'      : nodes,
                'size'       : len(nodes),
                'fraud_count': fraud_cnt,
                'fraud_pct'  : fraud_cnt / len(nodes) * 100,
            })
    rings.sort(key=lambda x: x['fraud_count'], reverse=True)
    return rings


def render_pyvis(G, height=600):
    """Render the NetworkX graph with PyVis and return HTML string."""
    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#0a0e1a",
        font_color="#e0e6f0",
        directed=False,
    )
    net.from_nx(G)
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "nodes": {
        "borderWidth": 2,
        "shadow": { "enabled": true, "color": "rgba(0,255,200,0.3)", "size": 10 }
      },
      "edges": {
        "smooth": { "type": "continuous" }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html',
                                     mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        return open(f.name, 'r', encoding='utf-8').read()


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🕸️ TRUSTNET // FRAUD NETWORK GRAPH</div>
        <div class="header-sub">► CONNECTED RING DETECTION · TRANSACTION CLUSTERING · BEHAVIOURAL LINKING</div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.markdown(f'<div class="alert-box">⚠ DATA ERROR: {e}<br>Run steps 1–5 first.</div>',
                    unsafe_allow_html=True)
        return

    fraud_total = int(df['Class'].sum())
    legit_total = int((df['Class'] == 0).sum())

    # Top stats
    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, len(df),        "TOTAL TXS"),
        (c2, fraud_total,    "FRAUD TXS"),
        (c3, legit_total,    "LEGIT TXS"),
        (c4, f"{fraud_total/len(df)*100:.2f}%", "FRAUD RATE"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{num}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Controls
    left, right = st.columns([1, 3])

    with left:
        st.markdown('<div class="section-title">⚙ GRAPH CONTROLS</div>', unsafe_allow_html=True)
        n_fraud      = st.slider("Fraud nodes",       20, min(150, fraud_total), 60, 10)
        time_window  = st.slider("Time window (s)",   600, 7200, 3600, 600)
        amount_tol   = st.slider("Amount tolerance %", 5, 50, 15, 5) / 100
        graph_height = st.slider("Graph height (px)", 400, 900, 600, 50)
        build_btn    = st.button("⚡ BUILD NETWORK")

        st.markdown('<div class="section-title" style="margin-top:1.5rem">ℹ LEGEND</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        🔴 &nbsp;RED NODE &nbsp;&nbsp;— Fraud transaction<br>
        🔵 &nbsp;BLUE NODE — Legit transaction<br>
        ─── RED EDGE &nbsp;— Fraud-Fraud link<br>
        ─── BLUE EDGE — Mixed link<br><br>
        Edges connect transactions with<br>
        similar TIME + AMOUNT — classic<br>
        fraud-ring signatures.
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">🕸 TRANSACTION NETWORK</div>', unsafe_allow_html=True)

        if 'graph_html' not in st.session_state or build_btn:
            with st.spinner("Building fraud network..."):
                G, combined = build_fraud_graph(df, n_fraud, time_window, amount_tol)
                rings = detect_fraud_rings(G, combined)
                st.session_state['graph_html'] = render_pyvis(G, graph_height)
                st.session_state['rings']      = rings
                st.session_state['G']          = G

        components.html(st.session_state['graph_html'], height=graph_height + 20, scrolling=False)

    # Fraud rings panel
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚨 DETECTED FRAUD RINGS</div>', unsafe_allow_html=True)

    rings = st.session_state.get('rings', [])
    if not rings:
        st.markdown('<div class="info-box">No multi-node fraud rings detected with current settings.<br>'
                    'Try increasing the time window or amount tolerance.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f"**{len(rings)} fraud ring(s) detected**", unsafe_allow_html=False)
        cols = st.columns(min(3, len(rings)))
        for i, ring in enumerate(rings[:6]):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="ring-card">
                    <div class="ring-id">◈ RING-{i+1:03d}</div>
                    <div class="ring-stats">
                        Nodes: {ring['size']} &nbsp;|&nbsp;
                        Fraud: {ring['fraud_count']} &nbsp;|&nbsp;
                        {ring['fraud_pct']:.0f}% fraud
                    </div>
                </div>""", unsafe_allow_html=True)

    # Network stats
    G = st.session_state.get('G')
    if G:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 NETWORK STATISTICS</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        for col, num, label in [
            (s1, G.number_of_nodes(), "TOTAL NODES"),
            (s2, G.number_of_edges(), "TOTAL EDGES"),
            (s3, nx.number_connected_components(G), "COMPONENTS"),
            (s4, len(rings), "FRAUD RINGS"),
        ]:
            col.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{num}</div>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()