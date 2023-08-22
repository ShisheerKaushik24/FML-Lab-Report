
def Recur_func(n):  
   if n == 1:  # Factorial of 1 is 1
       return n
   else:  
       return n*Recur_func(n-1)  # Using a factorial number nx(n-1)x...x1

# take input from the user  
num = int(input("Enter a number: "))  
# check is the number is negative  
if num < 0:  
   print("Sorry mate! the entered number {} is non-positive number" .format(num))  
elif num == 0:  
   print("The factorial of 0 is 1")  
else:  
   print("The factorial of {} is {}" .format(num,Recur_func(num)))