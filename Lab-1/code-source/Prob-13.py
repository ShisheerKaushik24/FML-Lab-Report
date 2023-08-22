
# Initialize the tot_sum variable
total_sum = 0

# Read the file to calculate the sum
with open("data.txt", "r") as file:
    for list in file:
        # Convert the list to a number
        number = float(list)
        total_sum += number

# Print the result
print("The sum of all numbers in the file is: {}" .format(total_sum))
