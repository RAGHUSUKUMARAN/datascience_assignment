import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from your directory
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\20 timeseries\Timeseries\exchange_rate.csv"
exchange_df = pd.read_csv(file_path, parse_dates=[0])

# Display a quick preview
print(exchange_df.head())

# Plot the time series
plt.figure(figsize=(12,6))
plt.plot(exchange_df['date'], exchange_df['Ex_rate'], label='USD to AUD Exchange Rate')
plt.title('Exchange Rate Over Time (USD → AUD)')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.grid(True)
plt.show()
