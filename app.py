import streamlit as st
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Bidirectional LSTM Workspace",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Midnight Cyber-Glass Dark Theme CSS Injector ---
st.markdown("""
<style>
    /* 1. Page Background Setup (Deep Midnight) */
    .stApp {
        background-color: #090B10 !important;
        background-image: radial-gradient(circle at 50% 0%, #161B2A 0%, #090B10 70%) !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* 2. Style EVERY Streamlit block container into a dark glass card */
    [data-testid="stVerticalBlock"] > div:has(div.custom-block),
    .stBlock {
        background-color: rgba(20, 25, 40, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        padding: 28px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(0, 242, 254, 0.15) !important; /* Subtle Neon Cyan Border */
        margin-bottom: 25px !important;
    }
    
    /* 3. Gradient Accent Header Layout */
    .title-container {
        text-align: center;
        padding: 30px 0;
        background: rgba(10, 15, 25, 0.8);
        border-radius: 16px;
        margin-bottom: 30px;
        border-bottom: 2px solid #00F2FE;
        box-shadow: 0 10px 30px rgba(0, 242, 254, 0.05);
    }
    .main-title {
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 50%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 2.8rem !important;
        margin: 0 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .sub-title {
        color: #94A3B8 !important;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* 4. Customizing Inputs and Inner Elements for Dark Mode */
    .stTextArea textarea {
        background-color: #0F141E !important;
        border: 1px solid #1E293B !important;
        color: #00F2FE !important;
        border-radius: 10px !important;
        font-size: 1.05rem !important;
        font-family: 'Courier New', Courier, monospace;
    }
    .stTextArea textarea:focus {
        border-color: #00F2FE !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.3) !important;
    }
    
    /* 5. Cyberpunk Button Theme */
    .stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
        color: #090B10 !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease-in-out !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.5) !important;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important;
    }

    /* 6. Dynamic Colored Badges for Tensors */
    .direction-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.8rem;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .forward-pass { background-color: rgba(0, 242, 254, 0.1) !important; color: #00F2FE !important; border: 1px solid #00F2FE !important; }
    .backward-pass { background-color: rgba(233, 69, 96, 0.1) !important; color: #E94560 !important; border: 1px solid #E94560 !important; }
    .concat-state { background-color: rgba(124, 58, 237, 0.1) !important; color: #B794F4 !important; border: 1px solid #7C3AED !important; }

    /* 7. Markdown & Sidebar Overrides */
    h1, h2, h3, h4 { color: #FFFFFF !important; }
    p, li { color: #CBD5E1 !important; }
    
    [data-testid="stSidebar"] {
        background-color: rgba(9, 11, 16, 0.95) !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: transparent !important; 
    }
    
    hr { border-color: #1E293B !important; }
</style>
""", unsafe_allow_html=True)

# --- Bidirectional LSTM Simulation Engine ---
def run_bilstm_simulation(text, speed):
    tokens = [t.strip() for t in text.split() if t.strip()]
    if not tokens:
        return None, None, None, 0.5, []
    
    num_tokens = len(tokens)
    h_forward = np.zeros((num_tokens, 2))
    h_backward = np.zeros((num_tokens, 2))
    
    lower_tokens = [t.lower().strip(".,!?\"'") for t in tokens]
    has_negation = any(neg in lower_tokens for neg in ['not', 'never', 'no', 'isnt', 'wasnt'])
    
    # Forward Pass
    current_f = np.zeros(2)
    for i in range(num_tokens):
        seed = sum(ord(c) for c in tokens[i]) % 100
        np.random.seed(seed)
        bias = 0.0
        clean_t = lower_tokens[i]
        if clean_t in ['excellent', 'love', 'perfect', 'amazing', 'masterpiece']:
            bias = 0.6
        elif clean_t in ['bad', 'boring', 'waste', 'disappointment', 'terrible']:
            bias = -0.1 if has_negation else -0.6
        current_f = np.tanh(0.7 * current_f + np.random.normal(bias, 0.3, 2))
        h_forward[i] = current_f

    # Backward Pass
    current_b = np.zeros(2)
    for i in reversed(range(num_tokens)):
        seed = (sum(ord(c) for c in tokens[i]) + 50) % 100
        np.random.seed(seed)
        bias = 0.0
        clean_t = lower_tokens[i]
        if clean_t in ['excellent', 'love', 'perfect', 'amazing', 'masterpiece']:
            bias = 0.6
        elif clean_t in ['bad', 'boring', 'waste', 'disappointment', 'terrible']:
            bias = -0.1 if has_negation else -0.6
        if clean_t == 'not' and i + 1 < num_tokens and lower_tokens[i+1] in ['bad', 'boring']:
            bias = 0.8 
        current_b = np.tanh(0.7 * current_b + np.random.normal(bias, 0.3, 2))
        h_backward[i] = current_b
        
    history = []
    for i in range(num_tokens):
        concatenated = np.concatenate([h_forward[i], h_backward[i]])
        history.append({
            "step": i + 1,
            "token": tokens[i],
            "forward": h_forward[i].copy(),
            "backward": h_backward[i].copy(),
            "concat": concatenated
        })
        
    final_context = np.mean(h_forward[-1]) + np.mean(h_backward[0])
    sentiment_score = 1 / (1 + np.exp(-final_context * 2.5))
    
    return h_forward[-1], h_backward[0], np.concatenate([h_forward[-1], h_backward[0]]), sentiment_score, history

# --- Premium Colorful Gradient Header Section ---
st.markdown("""
<div class="title-container">
    <div class="main-title">🔁 Bidirectional LSTM Workbench</div>
    <div class="sub-title">Supervised Deep Learning Sequence Representation Analysis</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.markdown("### 🔧 Model Options")
merge_strategy = st.sidebar.selectbox("Vector Merge Mode", ["Concatenation", "Summation", "Average"])
delay = st.sidebar.slider("Calculation Execution Delay (s)", 0.0, 1.0, 0.05)

# --- Column Matrix Grid Layout ---
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    # Anchor marker to hook custom CSS class styling onto this element container
    st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
    st.subheader("📥 Input Target Sequence")
    sample_text = st.text_area(
        "Enter raw text stream below:",
        value="Not a boring film, it was an amazing masterpiece with excellent pacing.",
        height=100
    )
    trigger_process = st.button("🚀 Analyze Dual-Direction Sequences", use_container_width=True)

with col_right:
    st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
    st.subheader("📊 Output Metrics Engine")
    
    if trigger_process and sample_text.strip():
        with st.spinner("Processing deep network nodes..."):
            f_vector, b_vector, combined_vector, sentiment, trace_log = run_bilstm_simulation(sample_text, delay)
            
            progress = st.progress(0)
            for pct in range(100):
                time.sleep(0.001)
                progress.progress(pct + 1)
            
            # Numeric Feature Metrics Row
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Processed Tokens", len(sample_text.split()))
            m_col2.metric("Forward Pass Size", f"{len(f_vector)}D")
            m_col3.metric("Backward Pass Size", f"{len(b_vector)}D")
            
       
            
            # High Contrast Evaluation Callouts
            st.markdown("#### **Classification Inference**")
            if sentiment >= 0.5:
                st.success(f"🟢 **POSITIVE SENTIMENT DETECTED** // Confidence: {sentiment:.2%}")
            else:
                st.error(f"🔴 **NEGATIVE SENTIMENT DETECTED** // Confidence: {(1 - sentiment):.2%}")
                
            
            
            st.markdown("#### **Unified Hidden Layer States Block**")
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.markdown("<span class='direction-badge forward-pass'>Forward State Vector</span>", unsafe_allow_html=True)
                st.code(f"{f_vector}")
            with v_col2:
                st.markdown("<span class='direction-badge backward-pass'>Backward State Vector</span>", unsafe_allow_html=True)
                st.code(f"{b_vector}")
                
            st.markdown("<span class='direction-badge concat-state'>Concatenated Bidirectional Payload Matrix</span>", unsafe_allow_html=True)
            st.code(f"{combined_vector}")
    else:
        st.info("💡 Write an input sentence on the left and trigger processing to populate computational vectors.")

# --- Full Width Conceptual Section ---
st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
st.subheader("💡 Context Dependency Breakdown")
st.write("""
In natural sequence modeling, evaluating words in isolation fails on context shifts. 
By utilizing **Bidirectional LSTM structures**, data streams concurrently from both ends (left-to-right and right-to-left). 
This enables elements like negation tags (*'Not'*) to instantly link with remote descriptors (*'boring'*), ensuring accuracy.
""")

# --- Step expansion elements matrix ---
if trigger_process and sample_text.strip() and 'trace_log' in locals():
    st.subheader("📋 Step Ledger Breakdown")
    for item in trace_log:
        with st.expander(label=f"Token Block Index {item['step']}: [{item['token']}]"):
            t_col1, t_col2, t_col3 = st.columns(3)
            t_col1.markdown("**Forward Pass Result:**")
            t_col1.code(f"{item['forward']}")
            t_col2.markdown("**Backward Pass Result:**")
            t_col2.code(f"{item['backward']}")
            t_col3.markdown("**Merged Vector Output:**")
            t_col3.code(f"{item['concat']}")