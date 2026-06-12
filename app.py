import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# --- 1. PAGE SETUP & DESIGN LAYOUT ---
st.set_page_config(
    page_title="PulseLSTM | Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DARK TERMINAL CSS (PURE WHITE TEXT FIX) ---
st.markdown("""
    <style>
    /* Global App Framework Elements */
    .stApp {
        background-color: #0e1117;
        color: #ffffff !important;
    }
    
    /* Clean Crisp Sidebar Structuring */
    section[data-testid="stSidebar"] {
        background-color: #1a1c23 !important;
        border-right: 1px solid #2d3748;
    }
    
    /* FORCE ALL SIDEBAR TEXT TO WHITE */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .st-emotion-cache-10trncp {
        color: #ffffff !important;
    }
    
    /* Elegant Title Typography Setup */
    .main-title {
        background: linear-gradient(90deg, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        letter-spacing: -1px;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-title {
        color: #ffffff;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    /* Minimalist Micro Cards for Visual Elements */
    .finance-card {
        background: #1a1c23;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom Indicator Metrics Blocks */
    .metric-box {
        background: #0e1117;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2d3748;
        border-left: 4px solid #ffffff;
    }
    .metric-lbl { color: #ffffff; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }
    .metric-val { color: #ffffff; font-size: 2.2rem; font-weight: 700; margin-top: 5px; }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #ffffff !important;
    }
    hr { border-color: #2d3748 !important; }
    
    /* Dataframe Text */
    .stDataFrame { color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. MAIN HEADER & TICKER SEARCH ---
st.markdown('<h1 class="main-title">PulseLSTM Terminal</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Deep Learning Sequential Market Analysis Engine</p>', unsafe_allow_html=True)

col_search, col_blank = st.columns([1, 2])
with col_search:
    ticker_symbol = st.text_input("🔍 Search Market Ticker", value="AAPL", help="Enter a valid Yahoo Finance ticker symbol (e.g., TSLA, MSFT, BTC-USD)")

# --- 4. CONTROL INTERFACE SIDEBAR PANEL ---
st.sidebar.markdown("### ⚙️ Hyperparameter Tuning")
st.sidebar.markdown("---")

lstm_units_1 = st.sidebar.slider("LSTM Neurons Layer A", 20, 100, 50, step=10)
lstm_units_2 = st.sidebar.slider("LSTM Neurons Layer B", 10, 50, 30, step=10)
dropout_rate = st.sidebar.slider("Dropout Regularization", 0.0, 0.4, 0.2, step=0.05)

st.sidebar.markdown("### ⚡ Optimization Specs")
training_epochs = st.sidebar.slider("Training Epochs", 5, 50, 15, step=5)
batch_size_value = st.sidebar.selectbox("Batch Computation Window", options=[16, 32, 64], index=1)

# --- 5. DATA FETCHING & PROCESSING ---
@st.cache_data(ttl=3600)
def fetch_sequential_market_logs(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2026-01-01", progress=False)
    return df

with st.spinner(f"Establishing secure connection to {ticker_symbol} market data..."):
    try:
        raw_data = fetch_sequential_market_logs(ticker_symbol)
        if raw_data.empty:
            st.error("Error: Received completely empty token payload check from source records.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to fetch market metrics for code symbol {ticker_symbol}: {e}")
        st.stop()

# Isolate Closing Data Matrices
close_dataset = raw_data[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_dataset)

# Establish Train-Test Sizing
training_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[0:int(training_len), :]

# Formulate Step Sliding Windows
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# --- 6. CORE TENSORFLOW LSTM NETWORK ---
@st.cache_resource
def compile_lstm_architecture(units_a, units_b, drop, epochs, batch_size, _X_train, _y_train):
    lstm_net = Sequential([
        LSTM(units_a, return_sequences=True, input_shape=(_X_train.shape[1], 1)),
        Dropout(drop),
        LSTM(units_b, return_sequences=False),
        Dropout(drop),
        Dense(25),
        Dense(1)
    ])
    lstm_net.compile(optimizer='adam', loss='mean_squared_error')
    lstm_net.fit(_X_train, _y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    return lstm_net

with st.spinner("Compiling Neural Tensors (Training LSTM)..."):
    trained_lstm = compile_lstm_architecture(
        lstm_units_1, lstm_units_2, dropout_rate, 
        training_epochs, batch_size_value, X_train, y_train
    )

# --- 7. EVALUATING VALIDATION SET ---
test_data = scaled_data[training_len - 60:, :]
X_test = []
y_test = close_dataset[training_len:, :]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model_predictions = trained_lstm.predict(X_test, verbose=0)
model_predictions = scaler.inverse_transform(model_predictions)
rmse_metric = np.sqrt(np.mean(((model_predictions - y_test) ** 2)))

# --- 8. TABBED WORKSPACE LAYOUT ---
tab1, tab2, tab3 = st.tabs(["📉 Forecast Engine", "🗃️ Market Data Explorer", "🧠 Network Architecture"])

with tab1:
    col_metric, col_chart = st.columns([1, 3], gap="large")
    
    with col_metric:
        st.markdown('<div class="finance-card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-lbl">Model RMSE Score</div>
            <div class="metric-val">${rmse_metric:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("A lower Root Mean Squared Error (RMSE) signifies tighter alignment with actual historical trajectories.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart:
        st.markdown('<div class="finance-card">', unsafe_allow_html=True)
        
        train_df = raw_data.iloc[:training_len]
        valid_df = raw_data.iloc[training_len:].copy()
        valid_df['Predictions'] = model_predictions

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'].values.flatten(), mode='lines', name='Historical Baseline', line=dict(color='#a1a1aa', width=1.5)))
        fig_trend.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Close'].values.flatten(), mode='lines', name='Actual Future Target', line=dict(color='#ffffff', width=2)))
        fig_trend.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Predictions'].values.flatten(), mode='lines', name='LSTM Forecast Path', line=dict(color='#38bdf8', width=2, dash='dash')))

        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
            xaxis=dict(title="", showgrid=False, tickfont=dict(color="white")),
            yaxis=dict(
                title=dict(text="Asset Valuation ($)", font=dict(color="white")), 
                gridcolor='#2d3748', 
                tickfont=dict(color="white")
            ),
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="white"))
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="finance-card">', unsafe_allow_html=True)
    st.subheader(f"Raw Historical Matrices: {ticker_symbol}")
    st.dataframe(raw_data.sort_index(ascending=False), use_container_width=True, height=400)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="finance-card">', unsafe_allow_html=True)
    st.subheader("LSTM Topology Map")
    st.write(f"""
    * **Input Layer Dimensions:** `({X_train.shape[1]}, 1)`
    * **Layer A (LSTM):** `{lstm_units_1} Units` (Returns sequences)
    * **Regularization A:** `Dropout({dropout_rate})`
    * **Layer B (LSTM):** `{lstm_units_2} Units` (Returns final sequence vector)
    * **Regularization B:** `Dropout({dropout_rate})`
    * **Dense Transition Layer:** `25 Nodes`
    * **Output Layer:** `1 Node (Linear)`
    """)
    st.markdown('</div>', unsafe_allow_html=True)
