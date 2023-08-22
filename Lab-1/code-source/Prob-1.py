
def sum_func(n):
    even_sum = []
    for i in n:
        if i%2==0:
            even_sum.append(i)
    return even_sum

sum_list = []

Elements = int(input('Update the required number of Elements: '))
for i in range(Elements):
    iter = int(input())
    sum_list.append(iter)

print('Provided list of numbers were: ', sum_list)

sum_var = sum_func(sum_list)

print('The summation of all even numbers in the provided list is: ',sum(sum_var))
