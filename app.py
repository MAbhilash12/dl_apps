import streamlit as st
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="BERT Contextual Encoder",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Neon Glassmorphism Dark UI CSS Injection ---
st.markdown("""
<style>
    /* 1. Deep Space/Cyber Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #16213e 50%, #0f1015 100%) !important;
        color: #e2e8f0 !important;
    }
    
    /* 2. Glassmorphism Cards */
    [data-testid="stVerticalBlock"] > div:has(div.custom-block),
    .stBlock {
        background: rgba(22, 33, 62, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        padding: 28px !important;
        border: 1px solid rgba(0, 242, 254, 0.2) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 25px !important;
    }
    
    /* 3. Glowing Gradient Header Panel */
    .title-container {
        text-align: center;
        padding: 30px 0;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 16px;
        margin-bottom: 30px;
        border-bottom: 2px solid #e94560;
        box-shadow: 0 10px 30px rgba(233, 69, 96, 0.1);
    }
    .main-title {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 3rem !important;
        margin: 0 !important;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .sub-title {
        color: #a0aec0 !important;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 10px;
        letter-spacing: 1px;
    }
    
    /* 4. Neon Inputs Text Area */
    .stTextArea textarea {
        background-color: rgba(15, 16, 21, 0.8) !important;
        border: 1px solid #e94560 !important;
        color: #00f2fe !important;
        border-radius: 10px !important;
        font-size: 1.05rem !important;
        font-family: 'Courier New', Courier, monospace;
    }
    .stTextArea textarea:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important;
    }
    
    /* 5. Cyberpunk Button */
    .stButton > button {
        background: linear-gradient(45deg, #e94560 0%, #fe0979 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.7) !important;
        background: linear-gradient(45deg, #fe0979 0%, #e94560 100%) !important;
    }

    /* 6. BERT Component Badges */
    .bert-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.8rem;
        margin-right: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .token-id { background-color: rgba(0, 242, 254, 0.1); color: #00f2fe; border: 1px solid #00f2fe; }
    .attention-head { background-color: rgba(233, 69, 96, 0.1); color: #e94560; border: 1px solid #e94560; }

    /* 7. Markdown overrides for dark mode */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    p, li {
        color: #cbd5e1 !important;
    }

    /* 8. Sidebar Dark Overrides */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 16, 21, 0.95) !important;
        border-right: 1px solid rgba(0, 242, 254, 0.2) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: transparent !important; 
    }
</style>
""", unsafe_allow_html=True)

# --- Simulation Engine for BERT Mechanism ---
def simulate_bert_encoder(text):
    """Simulates BERT tokenization, special tags injection, and attention weighting."""
    raw_words = [w.strip() for w in text.split() if w.strip()]
    if not raw_words:
        return [], 0.5
    
    # BERT explicitly prepends a [CLS] classification token and appends a [SEP] separator token
    tokens = ["[CLS]"] + raw_words + ["[SEP]"]
    num_tokens = len(tokens)
    
    # Generate mock Transformer attention weights matrix (scaled dot-product attention)
    np.random.seed(len(text))
    attention_matrix = np.random.uniform(0.05, 0.2, (num_tokens, num_tokens))
    
    # Highlight high-importance semantic links (e.g., matching negations to targets)
    lower_tokens = [t.lower().strip(".,!?\"'") for t in tokens]
    if "not" in lower_tokens:
        not_idx = lower_tokens.index("not")
        for i, tok in enumerate(lower_tokens):
            if tok in ["boring", "bad", "terrible", "excellent", "amazing"]:
                attention_matrix[not_idx, i] = 0.75
                attention_matrix[i, not_idx] = 0.65
                
    # Softmax normalize row values to represent legal probabilities summing to 1.0
    for i in range(num_tokens):
        exp_row = np.exp(attention_matrix[i] * 3)
        attention_matrix[i] = exp_row / np.sum(exp_row)
        
    # Calculate a final classification evaluation based on the [CLS] attention vector
    cls_vector = attention_matrix[0]
    positive_signals = ["amazing", "masterpiece", "excellent", "perfect", "good"]
    negative_signals = ["boring", "bad", "terrible", "waste", "poor"]
    
    base_sentiment = 0.5
    for idx, tok in enumerate(lower_tokens):
        if tok in positive_signals:
            base_sentiment += (cls_vector[idx] * 1.5)
        elif tok in negative_signals:
            base_sentiment -= (cls_vector[idx] * 1.5)
            
    if "not" in lower_tokens:
        # Flip sentiment if basic negation structures are linked
        base_sentiment = 1.0 - base_sentiment if base_sentiment < 0.5 else base_sentiment + 0.1

    final_score = np.clip(base_sentiment, 0.02, 0.98)
    return tokens, attention_matrix, final_score

# --- Centered Premium Header Panel ---
st.markdown("""
<div class="title-container">
    <div class="main-title">⚡ Neural Core: BERT Encoder</div>
    <div class="sub-title">Bidirectional Encoder Representations from Transformers — Deep Attention Matrix</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Sidebar Configuration Defaults ---
st.sidebar.markdown("### 🔧 Transformer Settings")
bert_variant = st.sidebar.selectbox("Model Core Type", ["bert-base-uncased (12-Layers)", "bert-large-uncased (24-Layers)"])
num_heads = st.sidebar.slider("Active Attention Heads", 4, 16, 12)

# --- Dual Column Layout Grid System ---
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
    st.subheader("📥 Input Target Sequence")
    sample_text = st.text_area(
        "Enter sentence to execute through transformer layers:",
        value="Not a boring film, it was an amazing masterpiece with excellent pacing.",
        height=110
    )
    trigger_process = st.button("🚀 Initialize Tensor Execution", use_container_width=True)

with col_right:
    st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
    st.subheader("📊 Output Metrics Engine")
    
    if trigger_process and sample_text.strip():
        with st.spinner("Tokenizing and executing multi-head attention arrays..."):
            tokens, attention_weights, sentiment_score = simulate_bert_encoder(sample_text)
            
            progress = st.progress(0)
            for pct in range(100):
                time.sleep(0.001)
                progress.progress(pct + 1)
            
            # Numeric Feature Metrics Row 
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("BERT Sub-tokens", len(tokens))
            m_col2.metric("Attention Matrix", f"{len(tokens)} × {len(tokens)}")
            m_col3.metric("Attention Heads", f"{num_heads} Heads")
            
            st.divider()
            
            # High Contrast Evaluation Callouts
            st.markdown("#### **Classification Inference (Via [CLS] Token Key)**")
            if sentiment_score >= 0.5:
                st.success(f"🟢 **POSITIVE SENTIMENT DETECTED** // Confidence Level: {sentiment_score:.2%}")
            else:
                st.error(f"🔴 **NEGATIVE SENTIMENT DETECTED** // Confidence Level: {(1 - sentiment_score):.2%}")
                
            st.divider()
            
            # Displaying Token Payload Arrays
            st.markdown("#### **BERT Token Embedding Map**")
            st.markdown("<span class='bert-badge token-id'>Processed Input Tensors</span>", unsafe_allow_html=True)
            st.code(f"{tokens}")
    else:
        st.info("💡 Awaiting Sequence. Enter text on the left and initialize the tensor engine to map attention.")

# --- Full Width Conceptual Section ---
st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
st.subheader("💡 How BERT Differs from LSTMs & Bi-LSTMs")
st.write("""
Unlike LSTMs or GRUs that read text word-by-word in a sequence, **BERT completely ditches recurrent loops**. 
It processes the entire text block simultaneously using the **Self-Attention Mechanism**. 
""")

b_col1, b_col2, b_col3 = st.columns(3)
with b_col1:
    st.markdown("**1. No Time Steps**")
    st.caption("BERT reads all tokens at once, making it incredibly fast to train on massive amounts of data across parallel GPUs.")
with b_col2:
    st.markdown("**2. True Bidirectional Context**")
    st.caption("Instead of running two independent loops, BERT's attention heads look at every word's relation to all surrounding words simultaneously.")
with b_col3:
    st.markdown("**3. Special Classification Token ([CLS])**")
    st.caption("The very first token `[CLS]` acts as a central hub, aggregating the context of the entire sequence to perform classification tasks.")

# --- Attention Matrix Weight Map Breakdown ---
if trigger_process and sample_text.strip() and 'tokens' in locals():
    st.subheader("📋 Contextual Attention Maps Matrix")
    st.write("Expand each token block below to view its statistical attention distribution weight profile across the rest of the sequence:")
    
    for idx, token in enumerate(tokens):
        with st.expander(label=f"Token Matrix Block {idx} ➔ [{token}]"):
            st.markdown(f"<span class='bert-badge attention-head'>Attention Vector Weights: [{token}]</span>", unsafe_allow_html=True)
            
            # Map attention weights directly back to target words for high readability
            token_weight_mapping = {tokens[i]: f"{attention_weights[idx][i]:.4f}" for i in range(len(tokens))}
            st.json(token_weight_mapping)