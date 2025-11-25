import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from src.predict import predict_next_batch, initialize_buffer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hull Tactical | Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px 25px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] label { color: #666666 !important; font-size: 14px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #1e3d59 !important; font-weight: 700; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #444444 !important; }
    
    h1, h2, h3 { color: #1e3d59; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER: LABELS ---
def get_position_label(val):
    if val <= 0.1: return "🛡️ Cash / Defensive"
    if val <= 0.8: return "⚖️ Under-Weight"
    if val <= 1.2: return "✅ Market Weight"
    if val <= 1.5: return "🚀 Aggressive"
    return "⚡ High Leverage"

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=80)
    st.markdown("## **Control Panel**")
    
    st.info(
        """
        **System Status:** Ready
        
        **Strategy:**
        Target Volatility: 15% Annualized.
        
        **Legend:**
        * **Blue Graph:** AI Allocation
        * **Red Graph:** Market Risk (Volatility)
        """
    )
    
    uploaded_file = st.file_uploader("Upload Market Data (CSV)", type=['csv'])
    run_btn = st.button("🚀 Run Prediction", type="primary")

# --- MAIN PAGE ---
st.title("📈 Hull Tactical Asset Allocation")
st.markdown("### AI-Driven Financial Time-Series Forecasting")

if uploaded_file is not None and run_btn:
    with st.spinner('Initializing Pipeline & Processing Data...'):
        initialize_buffer()
        
        try:
            # 1. Process
            test_data = pd.read_csv(uploaded_file)
            raw_allocations = predict_next_batch(test_data)
            
            # 2. Results DF
            results_df = test_data.copy()
            results_df['Allocation_Pct'] = [x * 100 for x in raw_allocations]
            results_df['Position_Type'] = [get_position_label(x) for x in raw_allocations]
            
            # Date handling
            if 'date_id' in results_df.columns:
                results_df['Date'] = results_df['date_id']
            else:
                results_df['Date'] = range(len(results_df))
            
            # --- METRICS ---
            st.markdown("#### 📊 Portfolio Snapshot")
            m1, m2, m3, m4 = st.columns(4)
            
            avg_alloc = results_df['Allocation_Pct'].mean()
            max_alloc = results_df['Allocation_Pct'].max()
            current_alloc = results_df['Allocation_Pct'].iloc[-1]
            active_days = (results_df['Allocation_Pct'] > 1.0).sum()
            
            m1.metric("Avg. Exposure", f"{avg_alloc:.1f}%")
            m2.metric("Max Leverage", f"{max_alloc:.1f}%")
            m3.metric("Current Position", f"{current_alloc:.1f}%", get_position_label(current_alloc/100))
            m4.metric("Active Days", f"{active_days} / {len(results_df)}")
            
            st.divider()
            
            # --- CHART 1: LEVERAGE (The Outcome) ---
            st.subheader("1️⃣ Dynamic Portfolio Allocation (The Strategy)")
            fig_alloc = go.Figure()
            fig_alloc.add_trace(go.Scatter(
                x=results_df['Date'], y=results_df['Allocation_Pct'],
                mode='lines', name='AI Strategy',
                line=dict(color='#0052cc', width=2),
                fill='tozeroy', fillcolor='rgba(0, 82, 204, 0.1)'
            ))
            # Thresholds
            fig_alloc.add_hline(y=100, line_dash="dot", line_color="gray", annotation_text="100% (Market Weight)")
            fig_alloc.add_hline(y=0, line_color="red", line_width=2, annotation_text="0% (Cash)")
            fig_alloc.add_hline(y=200, line_dash="dash", line_color="orange", annotation_text="200% (Max Leverage)")

            fig_alloc.update_layout(
                yaxis_title="Exposure (%)", xaxis_title="Time (Date ID)",
                template="plotly_white", height=400, hovermode="x unified"
            )
            st.plotly_chart(fig_alloc, use_container_width=True)
            
            # --- CHART 2: VOLATILITY (The Cause) ---
            # We look for 'V1' which is the raw volatility feature in input data
            vol_col = 'V1' if 'V1' in results_df.columns else None
            
            if vol_col:
                st.subheader("2️⃣ Underlying Market Volatility (The Risk Signal)")
                st.caption("When Volatility (Red) spikes, the AI automatically reduces Allocation (Blue).")
                
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=results_df['Date'], y=results_df[vol_col],
                    mode='lines', name='Market Volatility',
                    line=dict(color='#EF553B', width=2),
                    fill='tozeroy', fillcolor='rgba(239, 85, 59, 0.1)'
                ))
                
                fig_vol.update_layout(
                    yaxis_title="Volatility (V1)", xaxis_title="Time (Date ID)",
                    template="plotly_white", height=300, hovermode="x unified"
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # --- DATA TABLE ---
            st.subheader("📋 Execution Log")
            display_df = results_df[['Date', 'Allocation_Pct', 'Position_Type']].copy()
            display_df['Allocation_Pct'] = display_df['Allocation_Pct'].map('{:.2f}%'.format)
            
            st.dataframe(display_df, use_container_width=True)
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Analysis CSV", csv, "hull_predictions.csv", "text/csv")
            
        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            st.exception(e)

else:
    st.info("👈 Upload 'test.csv' or 'demo_market_cycle.csv' in the sidebar.")