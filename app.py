import streamlit as st
import numpy as np
import pandas as pd
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="GRU Interactive Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling Injection ---
st.markdown("""
<style>
    /* Modern minimalist container styling */
    .custom-card {
        background-color: var(--background-color);
        border-radius: 8px;
        padding: 20px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Sleek gradient text */
    .main-title {
        background: linear-gradient(135deg, #2563eb, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem !important;
        margin-bottom: 5px;
    }
    
    .subtitle {
        color: gray;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Metric overrides for cleaner look */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'final_hidden' not in st.session_state:
    st.session_state.final_hidden = None
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'trace_log' not in st.session_state:
    st.session_state.trace_log = None

# --- GRU Processing Simulator Engine ---
def run_gru_simulation(text, processing_speed):
    tokens = [t.strip() for t in text.split() if t.strip()]
    if not tokens:
        return None, 0.5, []
    
    h_t = np.zeros(3)
    history = []
    
    # Progress bar setup
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, token in enumerate(tokens):
        status_text.text(f"Processing token: '{token}'...")
        seed = sum(ord(char) for char in token) % 100
        np.random.seed(seed)
        
        r_t = np.random.uniform(0.1, 0.9, 3)
        z_t = np.random.uniform(0.1, 0.9, 3)
        
        clean_token = token.lower().strip(".,!?\"'")
        bias = 0.0
        if clean_token in ['fantastic', 'incredible', 'great', 'love', 'good', 'best', 'brilliant']:
            bias = 0.7
        elif clean_token in ['bad', 'terrible', 'worst', 'boring', 'waste', 'awful', 'poor']:
            bias = -0.7
            
        candidate = np.tanh(r_t * h_t + np.random.normal(bias, 0.2, 3))
        h_t = (1 - z_t) * h_t + z_t * candidate
        h_t = np.clip(h_t, -1.0, 1.0)
        
        history.append({
            "Step": idx + 1,
            "Token": token,
            "Reset Gate (r)": np.round(r_t, 3),
            "Update Gate (z)": np.round(z_t, 3),
            "Hidden State (h)": np.round(h_t.copy(), 3)
        })
        
        time.sleep(processing_speed)
        progress_bar.progress((idx + 1) / len(tokens))
        
    status_text.empty()
    progress_bar.empty()
    
    score = 1 / (1 + np.exp(-np.mean(h_t) * 3))
    return h_t, score, history

# --- App Header layout ---
st.markdown('<h1 class="main-title">🧠 GRU Sequence Processor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">An interactive workspace for exploring Gated Recurrent Unit temporal memory chains.</p>', unsafe_allow_html=True)

# --- Top Settings Panel ---
with st.expander("⚙️ Configuration & Hyperparameters", expanded=False):
    col1, col2, col3 = st.columns(3)
    max_len = col1.slider("Maximum Input Sequence Tokens", 5, 50, 25)
    delay = col2.slider("Sequential Delay (seconds)", 0.0, 1.0, 0.15)
    
    col3.markdown("**Quick Vocabulary**")
    col3.markdown("- **Reset Gate (r):** Drops historical data unrelated to the present.\n- **Update Gate (z):** Controls what historic context carries forward.")

# --- Tabbed Interface ---
tab1, tab2, tab3 = st.tabs(["🧪 Live Simulation", "📊 Step-by-Step Matrix", "📐 Architecture Overview"])

# --- TAB 1: Live Simulation ---
with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("📥 Sequence Input")
        sample_text = st.text_area(
            "Target text for sentiment analysis:",
            value="The story structure was fantastic, but the final pacing felt terrible.",
            height=130
        )
        
        if st.button("🚀 Process Sequence", type="primary", use_container_width=True):
            if sample_text.strip():
                truncated_text = " ".join(sample_text.split()[:max_len])
                with col_output:
                    h, s, t_log = run_gru_simulation(truncated_text, delay)
                    # Save to session state
                    st.session_state.final_hidden = h
                    st.session_state.sentiment = s
                    st.session_state.trace_log = t_log
                    st.session_state.simulation_run = True

    with col_output:
        st.subheader("📤 Evaluation Engine")
        
        if st.session_state.simulation_run:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            
            # Sentiment Banner
            score = st.session_state.sentiment
            if score >= 0.5:
                st.success(f"🟢 **POSITIVE SENTIMENT** — Confidence: {score:.2%}")
            else:
                st.error(f"🔴 **NEGATIVE SENTIMENT** — Confidence: {(1 - score):.2%}")
            
            st.divider()
            
            # Hidden State Metrics
            st.markdown("#### **Final Hidden Matrix Vector ($h_t$)**")
            m1, m2, m3 = st.columns(3)
            h_state = st.session_state.final_hidden
            m1.metric("Dimension 0", f"{h_state[0]:.4f}")
            m2.metric("Dimension 1", f"{h_state[1]:.4f}")
            m3.metric("Dimension 2", f"{h_state[2]:.4f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Enter a sequence and click 'Process Sequence' to see the network evaluate the data.")

# --- TAB 2: State Evolution Matrix ---
with tab2:
    st.subheader("📋 Token-by-Token State Evolution")
    if st.session_state.simulation_run:
        # Convert list of dicts to DataFrame for cleaner UI
        df = pd.DataFrame(st.session_state.trace_log)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Step": st.column_config.NumberColumn(format="%d"),
                "Token": st.column_config.TextColumn(width="medium"),
            }
        )
    else:
        st.warning("No data available. Run the simulation in the first tab to view the state matrix.")

# --- TAB 3: Mathematical Framework ---
with tab3:
    st.subheader("📐 Mathematical Operations Matrix")
    st.write("For each incoming vector token state step, the tensor changes are calculated across these standard functional bounds:")
    
    st.info("Where $\sigma$ represents the sigmoid activation function constraining data flows securely within a $0$ to $1$ range.")
    
    st.latex(r"""r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)""")
    st.latex(r"""z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)""")
    st.latex(r"""\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b_h)""")
    st.latex(r"""h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t""")