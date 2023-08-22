
# Given list of words
list_of_words = ['IBM-qiskit', 'Google-cirq', 'AWS-bracket', 'Azure-quantum', 'Qutip']

# Initialize an empty list to store word lengths
length_of_words = []

# Iterate through the words in the list and calculate the length of each word
for word in list_of_words:
    length_of_word = len(word)
    length_of_words.append(length_of_word)

print(length_of_words)
