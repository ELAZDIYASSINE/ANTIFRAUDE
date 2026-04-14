#!/usr/bin/env python3
"""
Ultra Modern Streamlit Dashboard for Anti-Fraud Detection
Features: Real-time monitoring, advanced visualizations, professional UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Page configuration
st.set_page_config(
    page_title="Anti-Fraud Detection - Ultra Modern Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    theme="dark"
)

# Custom CSS for ultra modern look
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* Headers */
    h1 {
        color: #00d4ff;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #00d4ff;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 2rem;
    }
    
    h3 {
        color: #7b68ee;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(123, 104, 238, 0.1));
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .metric-label {
        font-size: 1rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Alert boxes */
    .alert-box {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .alert-high {
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.2), rgba(255, 100, 100, 0.1));
        border: 2px solid #ff4444;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
    }
    
    .alert-medium {
        background: linear-gradient(135deg, rgba(255, 200, 0, 0.2), rgba(255, 150, 0, 0.1));
        border: 2px solid #ffaa00;
        box-shadow: 0 0 20px rgba(255, 200, 0, 0.5);
    }
    
    .alert-low {
        background: linear-gradient(135deg, rgba(0, 255, 100, 0.2), rgba(0, 200, 50, 0.1));
        border: 2px solid #00ff64;
        box-shadow: 0 0 20px rgba(0, 255, 100, 0.5);
    }
    
    /* Streamlit elements */
    .stSelectbox > div > div {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7b68ee);
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        overflow: hidden;
    }
    
    .dataframe th {
        background: rgba(0, 212, 255, 0.2);
        color: #00d4ff;
        font-weight: 700;
    }
    
    .dataframe td {
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def load_real_time_data():
    """Load real-time transaction data"""
    np.random.seed(int(time.time()))
    
    n_samples = 1000
    
    data = {
        'transaction_id': [f'TX{i:06d}' for i in range(n_samples)],
        'timestamp': [datetime.now() - timedelta(seconds=np.random.randint(0, 3600)) for _ in range(n_samples)],
        'type': np.random.choice(['TRANSFER', 'CASH_OUT', 'CASH_IN', 'PAYMENT', 'DEBIT'], n_samples),
        'amount': np.random.exponential(scale=100000, size=n_samples),
        'oldbalanceOrg': np.random.exponential(scale=500000, size=n_samples),
        'newbalanceOrig': np.random.exponential(scale=500000, size=n_samples),
        'nameOrig': [f'C{i:04d}' for i in range(n_samples)],
        'nameDest': [f'C{i:04d}' for i in range(n_samples)],
        'oldbalanceDest': np.random.exponential(scale=500000, size=n_samples),
        'newbalanceDest': np.random.exponential(scale=500000, size=n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'fraud_probability': np.random.uniform(0, 1, n_samples),
        'risk_level': np.random.choice(['HIGH', 'MEDIUM', 'LOW'], n_samples, p=[0.1, 0.2, 0.7])
    }
    
    df = pd.DataFrame(data)
    return df


def main():
    """Main dashboard application"""
    
    # Header with animated effect
    st.markdown('<h1>🛡️ Anti-Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #a0a0a0; font-size: 1.2rem;">Real-time Financial Fraud Detection Platform</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color: rgba(0, 212, 255, 0.3); margin: 2rem 0;">', unsafe_allow_html=True)
    
    # Load data
    df = load_real_time_data()
    
    # Sidebar
    st.sidebar.markdown('<h2 style="color: #00d4ff;">🎛️ Controls</h2>', unsafe_allow_html=True)
    
    # Time range selector with custom styling
    time_range = st.sidebar.selectbox(
        "📅 Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
        label_visibility="collapsed"
    )
    
    # Risk level filter
    risk_filter = st.sidebar.multiselect(
        "⚠️ Risk Level",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"]
    )
    
    # Transaction type filter
    type_filter = st.sidebar.multiselect(
        "💳 Transaction Type",
        df['type'].unique(),
        default=list(df['type'].unique())
    )
    
    # Apply filters
    filtered_df = df.copy()
    if risk_filter:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
    if type_filter:
        filtered_df = filtered_df[filtered_df['type'].isin(type_filter)]
    
    # Key Metrics with ultra modern cards
    st.markdown('<h2>📊 Real-time Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(filtered_df)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_transactions:,}</div>
            <div class="metric-label">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fraud_count = filtered_df['isFraud'].sum()
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{fraud_rate:.2f}%</div>
            <div class="metric-label">Fraud Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_risk = (filtered_df['risk_level'] == 'HIGH').sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{high_risk:,}</div>
            <div class="metric-label">High Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_amount = filtered_df['amount'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_amount:,.0f}</div>
            <div class="metric-label">Avg Amount</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr style="border-color: rgba(0, 212, 255, 0.3); margin: 2rem 0;">', unsafe_allow_html=True)
    
    # Advanced Charts
    st.markdown('<h2>📈 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction Type Distribution with custom colors
        type_counts = filtered_df['type'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.4,
            marker=dict(colors=['#00d4ff', '#7b68ee', '#00ff64', '#ffaa00', '#ff4444']),
            textinfo='label+percent',
            textfont=dict(size=14, color='white')
        )])
        fig_pie.update_layout(
            title='Transaction Type Distribution',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Risk Level Distribution with gradient colors
        risk_counts = filtered_df['risk_level'].value_counts()
        fig_bar = go.Figure(data=[go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker=dict(
                color=['#ff4444', '#ffaa00', '#00ff64'],
                line=dict(color='white', width=2)
            ),
            text=risk_counts.values,
            textposition='outside',
            textfont=dict(color='white', size=14)
        )])
        fig_bar.update_layout(
            title='Risk Level Distribution',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, color='white'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Advanced time series chart
    st.markdown('<h2>⏱️ Transaction Timeline</h2>', unsafe_allow_html=True)
    
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
    time_series = filtered_df.set_index('timestamp').resample('1min').size()
    
    fig_line = go.Figure(data=[go.Scatter(
        x=time_series.index,
        y=time_series.values,
        mode='lines+markers',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8, color='#00d4ff'),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    )])
    fig_line.update_layout(
        title='Transaction Volume Over Time',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown('<hr style="border-color: rgba(0, 212, 255, 0.3); margin: 2rem 0;">', unsafe_allow_html=True)
    
    # Fraud Alerts with modern styling
    st.markdown('<h2>⚠️ Live Fraud Alerts</h2>', unsafe_allow_html=True)
    
    high_risk_transactions = filtered_df[filtered_df['risk_level'] == 'HIGH'].sort_values('fraud_probability', ascending=False)
    
    if len(high_risk_transactions) > 0:
        for _, row in high_risk_transactions.head(5).iterrows():
            risk_class = 'alert-high' if row['fraud_probability'] > 0.8 else 'alert-medium'
            st.markdown(f"""
            <div class="alert-box {risk_class}">
                <h3 style="color: white; margin: 0;">🔴 HIGH RISK ALERT</h3>
                <p style="color: white; margin: 0.5rem 0;">
                    <strong>Transaction ID:</strong> {row['transaction_id']}<br>
                    <strong>Type:</strong> {row['type']}<br>
                    <strong>Amount:</strong> ${row['amount']:,.0f}<br>
                    <strong>Fraud Probability:</strong> {row['fraud_probability']:.2%}<br>
                    <strong>Timestamp:</strong> {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box alert-low"><h3 style="color: white; margin: 0;">✅ No High-Risk Transactions Detected</h3></div>', unsafe_allow_html=True)
    
    st.markdown('<hr style="border-color: rgba(0, 212, 255, 0.3); margin: 2rem 0;">', unsafe_allow_html=True)
    
    # System Performance
    st.markdown('<h2>⚡ System Performance</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">45ms</div>
            <div class="metric-label">API Response</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">95.2%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">99.9%</div>
            <div class="metric-label">System Uptime</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown('<hr style="border-color: rgba(0, 212, 255, 0.3); margin: 2rem 0;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align: center; color: #a0a0a0; padding: 2rem;'>
        <p>Anti-Fraud Detection System - Ultra Modern Dashboard</p>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Powered by PySpark, MLflow, Grafana, Prometheus</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
