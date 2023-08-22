
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
 
# Sample dataframe with 'Species' and 'age' columns
data = {'tree_Species': ['Pine', 'Olive', 'Beech', 'Yew', 'Baobab', 'Juniper', 'Redwood'],
        'age': [200, 150, 90, 100, 95, 120, 150]}
df = pd.DataFrame(data)

# Create a box plot
plt.figure(figsize=(9, 5))
sns.boxplot(x='tree_Species', y='age', data=df)

# Set plot title and labels
plt.title('Distribution of Ages for Different Species of Trees')
plt.xlabel('tree_Species')
plt.ylabel('Age')

# Show the plot
plt.show()
