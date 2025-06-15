import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setting up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Loading the dataset
df = pd.read_csv("city_transit_data_sample.csv")

# Converting timestamp columns to datetime
df['Scheduled_Arrival'] = pd.to_datetime(df['Scheduled_Arrival'])
df['Actual_Arrival'] = pd.to_datetime(df['Actual_Arrival'])
df['Date'] = df['Scheduled_Arrival'].dt.date
df['Hour'] = df['Scheduled_Arrival'].dt.hour

# Handling missing values (if any)
df = df.dropna(subset=['Delay_Minutes', 'Weather_Condition', 'Route', 'Station'])

# Time Series Analysis: Average delay by hour
hourly_delays = df.groupby('Hour')['Delay_Minutes'].mean()
plt.figure()
hourly_delays.plot(kind='line', marker='o', color='b')
plt.title('Average Delay by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Delay (Minutes)')
plt.xticks(range(0, 24))
plt.savefig('hourly_delays.png')
plt.close()

# Route-Based Analysis: Average delay by route
route_delays = df.groupby('Route')['Delay_Minutes'].mean().sort_values()
plt.figure()
route_delays.plot(kind='bar', color='green')
plt.title('Average Delay by Route')
plt.xlabel('Route')
plt.ylabel('Average Delay (Minutes)')
plt.savefig('route_delays.png')
plt.close()

# Station-Based Analysis: Average delay by station
station_delays = df.groupby('Station')['Delay_Minutes'].mean().sort_values()
plt.figure()
station_delays.plot(kind='bar', color='purple')
plt.title('Average Delay by Station')
plt.xlabel('Station')
plt.ylabel('Average Delay (Minutes)')
plt.savefig('station_delays.png')
plt.close()

# Weather Impact Analysis: Average delay by weather condition
weather_delays = df.groupby('Weather_Condition')['Delay_Minutes'].mean().sort_values()
plt.figure()
weather_delays.plot(kind='bar', color='orange')
plt.title('Average Delay by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Average Delay (Minutes)')
plt.savefig('weather_delays.png')
plt.close()

# Correlation Analysis: Create numeric proxy for weather conditions
weather_map = {'Clear': 0, 'Cloudy': 1, 'Rainy': 2, 'Snow': 3, 'Foggy': 4}
df['Weather_Score'] = df['Weather_Condition'].map(weather_map)

# Correlation matrix
corr_matrix = df[['Delay_Minutes', 'Weather_Score']].corr()
plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix: Delay vs. Weather Score')
plt.savefig('correlation_matrix.png')
plt.close()

# Delay Reason Analysis: Count of delays by reason
delay_reason_counts = df['Delay_Reason'].value_counts()
plt.figure()
delay_reason_counts.plot(kind='bar', color='red')
plt.title('Delay Reasons Count')
plt.xlabel('Delay Reason')
plt.ylabel('Count')
plt.savefig('delay_reasons.png')
plt.close()

# Summary statistics
summary = df.groupby(['Route', 'Weather_Condition'])['Delay_Minutes'].agg(['mean', 'count']).reset_index()
summary.to_csv('delay_summary.csv', index=False)

# Printing key insights
print("Key Insights:")
print(f"Average delay across all routes: {df['Delay_Minutes'].mean():.2f} minutes")
print(f"Route with highest average delay: {route_delays.idxmax()} ({route_delays.max():.2f} minutes)")
print(f"Weather condition with highest average delay: {weather_delays.idxmax()} ({weather_delays.max():.2f} minutes)")
print("\nSummary statistics saved to 'delay_summary.csv'")
print("Visualizations saved as PNG files: hourly_delays, route_delays, station_delays, weather_delays, correlation_matrix, delay_reasons")
