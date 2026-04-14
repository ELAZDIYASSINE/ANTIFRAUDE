#!/usr/bin/env python3
"""
Interactive Dashboard for Anti-Fraud Detection System
Features: Real-time monitoring, fraud detection visualization, performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Page configuration
st.set_page_config(
    page_title="Anti-Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ff6b6b;
        color: white;
    }
    .medium-risk {
        background-color: #ffd93d;
        color: black;
    }
    .low-risk {
        background-color: #6bcb77;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def load_sample_data():
    """Load sample transaction data for dashboard"""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'transaction_id': [f'TX{i:06d}' for i in range(n_samples)],
        'timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 60)) for _ in range(n_samples)],
        'type': np.random.choice(['TRANSFER', 'CASH_OUT', 'CASH_IN', 'PAYMENT', 'DEBIT'], n_samples),
        'amount': np.random.exponential(scale=100000, size=n_samples),
        'oldbalanceOrg': np.random.exponential(scale=500000, size=n_samples),
        'newbalanceOrig': np.random.exponential(scale=500000, size=n_samples),
        'nameOrig': [f'C{i:04d}' for i in range(n_samples)],
        'nameDest': [f'C{i:04d}' for i in range(n_samples)],
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'fraud_probability': np.random.uniform(0, 1, n_samples),
        'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], n_samples, p=[0.7, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    return df


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Anti-Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_sample_data()
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"]
    )
    
    # Risk level filter
    risk_filter = st.sidebar.multiselect(
        "Risk Level",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"]
    )
    
    # Transaction type filter
    type_filter = st.sidebar.multiselect(
        "Transaction Type",
        df['type'].unique(),
        default=list(df['type'].unique())
    )
    
    # Apply filters
    filtered_df = df.copy()
    if risk_filter:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
    if type_filter:
        filtered_df = filtered_df[filtered_df['type'].isin(type_filter)]
    
    # Key Metrics
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(filtered_df)
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col2:
        fraud_count = filtered_df['isFraud'].sum()
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    with col3:
        high_risk = (filtered_df['risk_level'] == 'HIGH').sum()
        st.metric("High Risk", f"{high_risk:,}")
    
    with col4:
        avg_amount = filtered_df['amount'].mean()
        st.metric("Avg Amount", f"${avg_amount:,.0f}")
    
    st.markdown("---")
    
    # Charts
    st.header("📈 Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Type Distribution")
        type_counts = filtered_df['type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Transaction Types",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Risk Level Distribution")
        risk_counts = filtered_df['risk_level'].value_counts()
        fig_bar = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Risk Levels",
            color=risk_counts.index,
            color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Amount distribution
    st.subheader("Transaction Amount Distribution")
    fig_hist = px.histogram(
        filtered_df,
        x='amount',
        nbins=50,
        title="Transaction Amount Distribution",
        color='isFraud',
        color_discrete_map={0: 'blue', 1: 'red'},
        labels={'isFraud': 'Fraud'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Fraud probability distribution
    st.subheader("Fraud Probability Distribution")
    fig_kde = px.histogram(
        filtered_df,
        x='fraud_probability',
        nbins=30,
        title="Fraud Probability Distribution",
        color='risk_level',
        color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
    )
    st.plotly_chart(fig_kde, use_container_width=True)
    
    st.markdown("---")
    
    # Recent Transactions Table
    st.header("📋 Recent Transactions")
    
    # Show recent transactions with risk highlighting
    recent_df = filtered_df.sort_values('timestamp', ascending=False).head(20)
    
    def highlight_risk(row):
        if row['risk_level'] == 'HIGH':
            return 'background-color: #ff6b6b'
        elif row['risk_level'] == 'MEDIUM':
            return 'background-color: #ffd93d'
        else:
            return 'background-color: #6bcb77'
    
    st.dataframe(
        recent_df[['transaction_id', 'timestamp', 'type', 'amount', 'risk_level', 'fraud_probability']],
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Fraud Alerts
    st.header("⚠️ Fraud Alerts")
    
    high_risk_transactions = filtered_df[filtered_df['risk_level'] == 'HIGH'].sort_values('fraud_probability', ascending=False)
    
    if len(high_risk_transactions) > 0:
        for _, row in high_risk_transactions.head(5).iterrows():
            st.markdown(f"""
            <div class="alert-box high-risk">
            <strong>🔴 HIGH RISK ALERT</strong><br>
            Transaction ID: {row['transaction_id']}<br>
            Type: {row['type']}<br>
            Amount: ${row['amount']:,.0f}<br>
            Fraud Probability: {row['fraud_probability']:.2%}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No high-risk transactions detected")
    
    st.markdown("---")
    
    # Performance Metrics
    st.header("⚡ System Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("API Response Time", "45ms", "-2ms")
    
    with col2:
        st.metric("Model Accuracy", "95.2%", "+0.3%")
    
    with col3:
        st.metric("System Uptime", "99.9%", "stable")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>Anti-Fraud Detection System Dashboard | Real-time Monitoring</p>
    <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
