import pandas as pd

# Create a dictionary with some sample data
data = {
    'Voltage': [5, 10, 15, 20, 25],
    'Current': [0.5, 1.0, 1.5, 2.0, 2.5],
    'Resistance': [10, 10, 10, 10, 10]
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('circuit_data.csv', index=False)

print("Dataset created and saved to 'circuit_data.csv'")