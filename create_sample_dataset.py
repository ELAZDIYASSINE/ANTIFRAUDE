import pandas as pd
import numpy as np
import os

def create_sample_paysim_dataset():
    """Create a sample PaySim-like dataset for demonstration"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 10000
    
    # Create realistic sample data
    data = {
        'step': np.random.randint(1, 744, n_samples),
        'type': np.random.choice(['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'], n_samples, p=[0.35, 0.34, 0.22, 0.08, 0.01]),
        'amount': np.round(np.random.exponential(scale=100000, size=n_samples), 2),
        'nameOrig': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_samples)],
        'oldbalanceOrg': np.round(np.random.exponential(scale=500000, size=n_samples), 2),
        'newbalanceOrig': np.round(np.random.exponential(scale=400000, size=n_samples), 2),
        'nameDest': [f'M{np.random.randint(1000000, 9999999)}' for _ in range(n_samples)],
        'oldbalanceDest': np.round(np.random.exponential(scale=300000, size=n_samples), 2),
        'newbalanceDest': np.round(np.random.exponential(scale=350000, size=n_samples), 2),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),  # ~0.2% fraud rate
        'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.9995, 0.0005])  # ~0.05% flagged
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure logical consistency
    df.loc[df['type'] == 'CASH_IN', 'newbalanceOrig'] = df.loc[df['type'] == 'CASH_IN', 'oldbalanceOrg'] + df.loc[df['type'] == 'CASH_IN', 'amount']
    df.loc[df['type'] == 'CASH_OUT', 'newbalanceOrig'] = df.loc[df['type'] == 'CASH_OUT', 'oldbalanceOrg'] - df.loc[df['type'] == 'CASH_OUT', 'amount']
    df.loc[df['type'] == 'TRANSFER', 'newbalanceOrig'] = df.loc[df['type'] == 'TRANSFER', 'oldbalanceOrg'] - df.loc[df['type'] == 'TRANSFER', 'amount']
    df.loc[df['type'] == 'TRANSFER', 'newbalanceDest'] = df.loc[df['type'] == 'TRANSFER', 'oldbalanceDest'] + df.loc[df['type'] == 'TRANSFER', 'amount']
    
    # Save to CSV
    filename = "data/PS_20174392719_1491204439457_log.csv"
    df.to_csv(filename, index=False)
    
    print(f"Sample dataset created: {filename}")
    print(f"Shape: {df.shape}")
    print(f"Fraud cases: {df['isFraud'].sum()} ({df['isFraud'].mean():.4%})")
    print(f"Flagged fraud cases: {df['isFlaggedFraud'].sum()} ({df['isFlaggedFraud'].mean():.4%})")
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    create_sample_paysim_dataset()
