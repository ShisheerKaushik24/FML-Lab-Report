
import matplotlib.pyplot as plt

# Initiate a random data for the closing prices over ten days
days = [1,2,3,4,5,6,7,8,9,10]
closing_prices = [75,68,80,98,65,75,78,82,86,93]

# Creating a line plot
plt.figure(figsize=(6, 4))  # Set the figure size
plt.plot(days, closing_prices, marker='o', linestyle='-', color='b', label='Closing Price')

# labels and title
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.title('Stock Closing Price Trend Over Ten Days')
plt.xticks(days)  
plt.legend()  # legend

plt.show()
