
import pandas as pd

def fill_missing_with_mean(df, inplace=False):
    # mean for each column
    column_means = df.mean()
    
    # Fill missing values in each column with the corresponding mean
    if inplace:
        df.fillna(column_means, inplace=True)
        return None
    else:
        df_filled = df.fillna(column_means)
        return df_filled

# sample DataFrame with missing values

data = pd.DataFrame({"A":[12, 4, 5, None, 1], 
                   "B":[None, 2, 54, 3, None], 
                   "C":[20, 16, None, 3, 8], 
                   "D":[14, 3, None, None, 6]}) 

df = pd.DataFrame(data)

# Call the defined function
filled_df = fill_missing_with_mean(df)

print("\nDataFrame:")
print(df)
print("\nDataFrame filled with means in the missing value:")
print(filled_df)
