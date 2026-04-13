import requests
import os

def download_paysim_dataset():
    """Download the PaySim dataset from a reliable source"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Try multiple sources
    sources = [
        "https://raw.githubusercontent.com/EdgarLopezPhD/PaySim/master/data/PS_20174392719_1491204439457_log.csv",
        "https://github.com/EdgarLopezPhD/PaySim/raw/master/data/PS_20174392719_1491204439457_log.csv",
        "https://storage.googleapis.com/learnjs-data/model-builder/PS_20174392719_1491204439457_log.csv"
    ]
    
    for i, url in enumerate(sources):
        print(f"Trying source {i+1}: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if we got actual CSV data
            if response.text.startswith('step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud'):
                filename = "data/PS_20174392719_1491204439457_log.csv"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"Successfully downloaded dataset to {filename}")
                print(f"File size: {len(response.text)} characters")
                return True
            else:
                print(f"Source {i+1} returned invalid data (not CSV)")
                
        except Exception as e:
            print(f"Source {i+1} failed: {e}")
    
    print("All sources failed. Dataset could not be downloaded.")
    return False

if __name__ == "__main__":
    download_paysim_dataset()
