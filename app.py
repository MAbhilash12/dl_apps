import streamlit as st
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="LSTM Sequence Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Cyber/Dark Terminal UI CSS Injection ---
st.markdown("""
<style>
    /* 1. Dark Theme Background with Pure White Text */
    .stApp {
        background-color: #0d1117 !important;
        color: #ffffff !important; /* Brighter global text */
    }
    
    /* Global text overrides for markdown */
    p, li, .stMarkdown {
        color: #f0f6fc !important; /* Ultra-bright text for standard paragraphs */
        font-size: 1.05rem;
    }
    
    /* 2. Custom Glowing Cards */
    [data-testid="stVerticalBlock"] > div:has(div.custom-card),
    .stBlock {
        background-color: #161b22 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        border: 1px solid #30363d !important;
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.05) !important;
        margin-bottom: 20px !important;
    }
    
    /* 3. Futuristic Header */
    .cyber-header {
        text-align: center;
        padding: 40px 0;
        background: radial-gradient(circle, rgba(23,34,52,1) 0%, rgba(13,17,23,1) 100%);
        border-bottom: 1px solid #58a6ff;
        margin-bottom: 40px;
        border-radius: 0 0 20px 20px;
    }
    .cyber-title {
        color: #79c0ff !important; /* Brighter neon blue */
        font-weight: 900 !important;
        font-size: 2.8rem !important; /* Slightly smaller to fit the longer text */
        font-family: 'Courier New', Courier, monospace;
        text-shadow: 0 0 18px rgba(121, 192, 255, 0.6);
        margin: 0 !important;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .cyber-subtitle {
        color: #e6edf3 !important; /* Brighter subtitle */
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    /* 4. Terminal Text Area */
    .stTextArea textarea {
        background-color: #010409 !important;
        border: 1px solid #30363d !important;
        color: #ffffff !important; /* Pure white input text */
        border-radius: 8px !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }
    .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 12px rgba(88, 166, 255, 0.4) !important;
    }
    
    /* 5. Neon Button */
    .stButton > button {
        background: #21262d !important;
        color: #79c0ff !important; /* Brighter text on button */
        border: 1px solid #58a6ff !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        border-radius: 8px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: #58a6ff !important;
        color: #0d1117 !important;
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.8) !important;
    }

    /* 6. Status Badges */
    .terminal-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: bold;
        font-size: 0.95rem;
        margin-bottom: 10px;
    }
    .c-state { background-color: rgba(63, 185, 80, 0.15); color: #56d364; border: 1px solid #56d364; }
    .h-state { background-color: rgba(210, 153, 34, 0.15); color: #e3b341; border: 1px solid #e3b341; }
    
    /* Override markdown headers to be brighter */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    /* 7. Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d !important;
    }
    /* Ensures the inner sidebar container also matches */
    [data-testid="stSidebar"] > div:first-child {
        background-color: transparent !important; 
    }
</style>
""", unsafe_allow_html=True)

# --- Mock LSTM Recurrent Core Engine ---
def simulate_lstm_sentiment(text):
    """Simulates LSTM sequence accumulation with Cell States & Hidden States."""
    words = [w.strip() for w in text.split() if w.strip()]
    if not words:
        return 0.5, 0, []
    
    c_t = np.zeros(3)
    h_t = np.zeros(3)
    word_traces = []
    
    for idx, word in enumerate(words):
        seed = sum(ord(char) for char in word) % 100
        np.random.seed(seed)
        
        clean_word = word.lower().strip(".,!?\"'")
        sentiment_pull = 0.0
        if clean_word in ['amazing', 'good', 'love', 'masterpiece', 'great', 'excellent', 'beautiful']:
            sentiment_pull = 0.5
        elif clean_word in ['bad', 'boring', 'terrible', 'worst', 'waste', 'horrible']:
            sentiment_pull = -0.5
            
        forget_gate = np.random.uniform(0.6, 0.95, 3) 
        input_gate = np.random.uniform(0.2, 0.8, 3)
        output_gate = np.random.uniform(0.3, 0.8, 3)
        
        candidate_cell = np.tanh(np.random.normal(sentiment_pull, 0.2, 3))
        c_t = (forget_gate * c_t) + (input_gate * candidate_cell)
        h_t = output_gate * np.tanh(c_t)
        
        word_traces.append({
            "index": idx + 1,
            "word": word,
            "cell_state": c_t.copy(),
            "hidden_state": h_t.copy()
        })
        
    final_score = 1 / (1 + np.exp(-np.mean(h_t) * 4))
    return final_score, len(words), word_traces

# --- Header Panel ---
st.markdown("""
<div class="cyber-header">
    <h1 class="cyber-title">Sentiment Analysis Using LSTM</h1>
    <div class="cyber-subtitle">Deep Sequence Analysis Protocol</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Parameter Selection ---
st.sidebar.markdown("### ⚙️ Model Hyperparameters")
st.sidebar.markdown("Configure the LSTM architecture settings below:")

# Architecture options
st.sidebar.markdown("#### 🏗️ Architecture")
vocab_size = st.sidebar.selectbox("Embedding Vector Space", ["10,000 dimensions", "20,000 dimensions", "50,000 dimensions"], index=1)
embedding_dim = st.sidebar.selectbox("Embedding Output Dimension", [64, 128, 256, 512], index=1)
lstm_units = st.sidebar.slider("Hidden State Units", 32, 512, 128, step=32)
lstm_layers = st.sidebar.slider("Number of LSTM Layers", 1, 3, 1)

# Regularization options
st.sidebar.markdown("#### 🛡️ Regularization")
dropout_rate = st.sidebar.slider("Network Dropout", 0.0, 0.8, 0.2, step=0.1)
recurrent_dropout = st.sidebar.slider("Recurrent Dropout", 0.0, 0.5, 0.0, step=0.1)

st.sidebar.divider()
st.sidebar.caption("System Status: MODEL INITIALIZED & ONLINE")

# --- Main Input Section (Centered) ---
st.markdown('<div class="custom-card"></div>', unsafe_allow_html=True)
user_input = st.text_area(
    "INPUT SEQUENCE STRING:",
    value="The cinematic score was an amazing masterpiece, but some parts felt a bit boring.",
    height=100
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_analysis = st.button("Initialize Sequence Analysis")

st.divider()

# --- Tabbed Output Dashboard ---
if run_analysis and user_input.strip():
    with st.spinner("Compiling sequence and propagating through LSTM gates..."):
        score, token_count, execution_history = simulate_lstm_sentiment(user_input)
        
        # Cyber progress bar
        progress = st.progress(0)
        for step in range(100):
            time.sleep(0.001)
            progress.progress(step + 1)
        
        # Organize output into clean tabs
        tab1, tab2, tab3 = st.tabs(["📊 Inference Results", "🗄️ Step-by-Step Matrix Log", "📐 Architecture Specs"])
        
        with tab1:
            st.markdown("### Model Diagnostics")
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Processed Tokens", f"{token_count} units")
            kpi2.metric("Network Depth", f"{lstm_layers} Layers x {lstm_units} Units")
            kpi3.metric("State Retention", f"{(1 - dropout_rate):.0%}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Classification Block
            if score >= 0.5:
                st.success(f"**POSITIVE SENTIMENT DETECTED** // Confidence Metric: {score:.2%}")
            else:
                st.error(f"**NEGATIVE SENTIMENT DETECTED** // Confidence Metric: {(1 - score):.2%}")
                
            if execution_history:
                st.markdown("### Terminal Vectors")
                last_step = execution_history[-1]
                st.markdown("<div class='terminal-badge c-state'>C_t // FINAL_CELL_STATE</div>", unsafe_allow_html=True)
                st.code(f"{last_step['cell_state']}")
                st.markdown("<div class='terminal-badge h-state'>H_t // FINAL_HIDDEN_STATE</div>", unsafe_allow_html=True)
                st.code(f"{last_step['hidden_state']}")

        with tab2:
            st.markdown("### Sequence Execution Trace")
            st.caption("Expand indices to view vector transformations per token.")
            for log_item in execution_history:
                with st.expander(f"INDEX [{log_item['index']:02d}] :: TOKEN: '{log_item['word']}'"):
                    t_col1, t_col2 = st.columns(2)
                    t_col1.markdown("**Memory Accumulation ($C_t$):**")
                    t_col1.code(f"{log_item['cell_state']}")
                    t_col2.markdown("**Context Output ($H_t$):**")
                    t_col2.code(f"{log_item['hidden_state']}")

        with tab3:
            st.markdown("### Long Short-Term Memory Dynamics")
            st.write("""
            Standard Recurrent Networks suffer from **vanishing gradients** over long temporal dependencies. 
            An **LSTM layer** resolves this via a continuous internal manifold called the **Cell State ($C_t$)**. 
            At every sequence step, three distinct non-linear gates modulate information flow:
            """)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.info("**1. Forget Gate ($f_t$)**\n\nAttenuates outdated contextual matrices.")
            with m_col2:
                st.success("**2. Input Gate ($i_t$)**\n\nIntegrates incoming token payloads.")
            with m_col3:
                st.warning("**3. Output Gate ($o_t$)**\n\nProjects updated metrics into Hidden State ($h_t$).")
else:
    if not run_analysis:
        st.info("Awaiting system input. Enter a sequence and initialize analysis to view internal state vectors.")

