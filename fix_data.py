import pandas as pd
import os

# Load main data
df = pd.read_csv('outputs/processed_data.csv')
print(f" Loaded: {len(df):,} rows")

# Save with correct name
df.to_csv('powerbi/powerbi_dashboard_data.csv', index=False)
print(f" Saved: powerbi/powerbi_dashboard_data.csv")