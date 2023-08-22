
import pandas as pd

# Sample DataFrame
sample_data = {'Sales': [52050, 75502, 62090, 81280],
        'Expenses': [60000, 48800, 49000, 77700]}
df = pd.DataFrame(sample_data)

# Used assign() method to create a new column 'Profit'
new_df = df.assign(Profit=df['Sales'] - df['Expenses'])

print(new_df)
