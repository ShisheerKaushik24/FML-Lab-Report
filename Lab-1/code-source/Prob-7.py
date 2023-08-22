
import pandas as pd

# DataFrame
data = {
        'person': ['Olivia', 'Bobby', 'Casper', 'Davin', 'Ava', 'Robert', 'Agusthya', 'Sopia', 'Isabella', 'Martha'],
        'gender': ['female', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'female', 'female'],
        'age': [31, 25, 45, 32, 28, 35, 20, 15, 50, 30],
        }

df = pd.DataFrame(data)

# Filter the DataFrame using Query string
filt_df = df.query('age > 30 and gender == "female"')

print('Below is a list of Females above 30 years of age : {}' .format(filt_df))
