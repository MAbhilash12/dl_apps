import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# 1. PAGE SETUP & NORD THEME UI (CUSTOM CSS)
# ==============================================================================
st.set_page_config(
    page_title="GPT Architecture Studio", 
    page_icon="❄️", 
    layout="centered"
)

# Implementing the Arctic "Nord" Developer Theme
st.markdown(
    """
    <style>
    /* 1. Global Background (Polar Night) */
    .stApp {
        background-color: #2E3440 !important; 
        color: #ECEFF4 !important; /* Snow Storm text */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* 2. Main Headers - Frost Cyan */
    h1, h2, h3 {
        color: #88C0D0 !important; 
        font-weight: 700 !important;
        letter-spacing: 0.5px;
    }
    
    /* 3. Content Cards - Elevated Polar Night */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #3B4252;
        border: 1px solid #434C5E;
        border-radius: 12px;
        padding: 4px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* 4. Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #242933 !important;
        border-right: 1px solid #3B4252;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #D8DEE9 !important;
    }
    
    /* 5. Input Box styling - Muted Gray with Blue Focus */
    .stTextInput input, .stNumberInput input {
        border-radius: 6px !important;
        border: 1px solid #4C566A !important;
        background-color: #2E3440 !important;
        color: #ECEFF4 !important;
        transition: all 0.2s ease;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #81A1C1 !important;
        box-shadow: 0 0 0 2px rgba(129, 161, 193, 0.3) !important;
        background-color: #3B4252 !important;
    }

    /* 6. Training Button - Aurora Green */
    .train-btn > div > button {
        background-color: #A3BE8C !important;
        color: #2E3440 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        height: 44px;
        transition: all 0.2s ease;
    }
    .train-btn > div > button:hover {
        background-color: #8FBCBB !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(143, 188, 187, 0.4) !important;
    }
    
    /* 7. Generation Button - Aurora Purple */
    .gen-btn > div > button {
        background-color: #B48EAD !important;
        color: #2E3440 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        height: 48px;
        transition: all 0.2s ease;
        margin-top: 10px;
    }
    .gen-btn > div > button:hover {
        filter: brightness(1.1);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(180, 142, 173, 0.4) !important;
    }
    
    /* 8. Terminal Output Panel - Minimalist Console */
    .terminal-container {
        background-color: #242933;
        border: 1px solid #434C5E;
        border-left: 4px solid #88C0D0;
        padding: 20px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 14.5px;
        line-height: 1.6;
        color: #EBCB8B; /* Aurora Yellow for generated text */
        white-space: pre-wrap;
    }
    
    /* General Typography Fixes */
    p, li, label, .stMarkdown {
        color: #E5E9F0 !important;
    }
    hr {
        border-color: #4C566A !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 2. MINI-GPT PYTORCH ARCHITECTURE
# ==============================================================================
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size) 

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================================================================
# 3. CORPUS DATASET SETUP
# ==============================================================================
training_corpus = """
Once upon a time, there was a tiny artificial intelligence model. 
The little model wanted to speak human language correctly. It tried to learn every day.
Once upon a time, the model worked hard. It processed tokens and adjusted weights. 
Suddenly, after training, it stopped generating gibberish. It spoke clearly and made sense!
Once upon a time, everyone celebrated the successful training of the tiny model.
"""

chars = sorted(list(set(training_corpus + " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n")))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi.get(c, 0) for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ==============================================================================
# 4. STREAMLIT FRONTEND & ENGINE RETENTION CONTROLLER
# ==============================================================================
st.title("GPT Architecture Studio")
st.write("Train and visualize an autoregressive Transformer language model natively inside a sleek Nord interface.")


# FIXED STRUCTURAL PROPERTIES
n_embd = 64
n_head = 4
n_layer = 3
block_size = 16

st.sidebar.markdown("### 🛠️ Neural Architecture")
st.sidebar.caption("Structural dimensions optimized for browser execution.")
st.sidebar.markdown(f"""
* **Embedding Space:** `{n_embd}` channels
* **Attention Heads:** `{n_head}`
* **Transformer Layers:** `{n_layer}`
* **Context Bounds:** `{block_size}` tokens
""")



st.sidebar.markdown("### 🎛️ Inference Settings")
max_tokens = st.sidebar.slider("Tokens to Output", 10, 300, 120)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.5, 0.6)

# SESSION RETENTION ENGINE FIX
if 'gpt_model' not in st.session_state:
    st.session_state.gpt_model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size, device).to(device)
    st.session_state.has_trained_successfully = False
    st.session_state.cumulative_steps = 0

# --- CARD 1: TRAINING ---
st.subheader("1. Optimize Model Weights")
st.caption("Feed the internal text database into your model matrices to train the attention map path vectors.")

col1, col2 = st.columns([1, 1])
with col1:
    epochs = st.number_input("Steps to Train", min_value=200, max_value=3000, value=1200, step=100)
    
with col2:
    st.markdown("<br>", unsafe_allow_html=True) # Alignment spacing
    st.markdown('<div class="train-btn">', unsafe_allow_html=True)
    if st.button("🚀 Run Neural Training Loop", use_container_width=True):
        data_tensor = torch.tensor(encode(training_corpus), dtype=torch.long)
        optimizer = torch.optim.AdamW(st.session_state.gpt_model.parameters(), lr=1e-3)
        
        loss_progress = st.progress(0)
        status_text = st.empty()
        chart_slot = st.empty()
        chart_data = []
        
        st.session_state.gpt_model.train()
        for step in range(epochs):
            xb, yb = get_batch(data_tensor, batch_size=16, block_size=block_size)
            logits, loss = st.session_state.gpt_model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0 or step == epochs - 1:
                loss_progress.progress((step + 1) / epochs)
                status_text.caption(f"Step {step}/{epochs} — Matrix Loss: `{loss.item():.4f}`")
                chart_data.append(loss.item())
                chart_slot.line_chart(chart_data)
                
        st.session_state.gpt_model.eval()
        st.session_state.has_trained_successfully = True
        st.session_state.cumulative_steps += epochs
        st.success(f"🎉 Model weights locked! Total loops run: {st.session_state.cumulative_steps}")
    st.markdown('</div>', unsafe_allow_html=True)



# --- CARD 2: GENERATION ---
st.subheader("2. Autoregressive Text Generation")
st.caption("Provide a context sequence to generate predictive text based on the trained weights.")

user_prompt = st.text_input("Enter your starting story prompt:", value="Once upon a time")

st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
if st.button("✨ Execute Generation Pipeline", use_container_width=True):
    if not st.session_state.has_trained_successfully:
        st.error("🛑 Safety Block: You cannot generate text yet! Please run the training loop in Step 1 first.")
    elif not user_prompt.strip():
        st.warning("Please supply an input sequence context first.")
    else:
        clean_prompt = "".join([c for c in user_prompt if c in stoi])
        if not clean_prompt:
            clean_prompt = "Once upon a time"
            
        with st.spinner("Decoding persistent attention maps..."):
            context_tokens = encode(clean_prompt)
            x = torch.tensor([context_tokens], dtype=torch.long, device=device)
            
            with torch.no_grad():
                generated_indices = st.session_state.gpt_model.generate(x, max_new_tokens=max_tokens, temperature=temperature)
                output_text = decode(generated_indices[0].tolist())
            
            st.markdown("### 🖥️ Studio Output Stream:")
            st.markdown(f'<div class="terminal-container">{output_text}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)