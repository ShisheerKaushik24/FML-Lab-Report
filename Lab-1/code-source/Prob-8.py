
import pandas as pd

# Sample DataFrame with population data for different centuries
data = {
    'Country': ['Germany', 'Poland', 'Russia'],
    '19th Century': [1200000, 560000, 1005000],
    '20th Century': [25000000, 51000000, 78000000],
    '21st Century': [175000000, 35000000, 80000000]
}

df = pd.DataFrame(data)

# calculate the average for each row
def calculate_average(row):
    # Exclude the 'Country' column
    population_values = row[1:]
    return population_values.mean()

# Apply the custom function to each row
df['Average Population'] = df.apply(calculate_average, axis=1)

print(df[['Country', 'Average Population']])

