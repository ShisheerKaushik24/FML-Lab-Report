import pandas as pd

# Define the CSV file path
csv_file = "data.csv"

##Read the CSV file into a pandas DataFrame
training_data = pd.read_csv(csv_file)

# Convert the DataFrame to a numpy array
training_data = training_data.values

# Now, 'df' is a DataFrame containing your CSV data
print(training_data)