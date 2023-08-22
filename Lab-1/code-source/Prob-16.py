
import matplotlib.pyplot as plt

# data with scores for each subject w.r.t every student
student_scores = {
    'Rajan': {'Algebra': 80, 'Geometry': 75, 'Calculus': 90},
    'Samar': {'Algebra': 88, 'Geometry': 72, 'Calculus': 78},
    'Doma': {'Algebra': 90, 'Geometry': 90, 'Calculus': 65},
    'Ludwic': {'Algebra': 75, 'Geometry': 78, 'Calculus': 82},
    'Max': {'Algebra': 73, 'Geometry': 85, 'Calculus': 70}
}

# Initialize a dictionary
average_scores = {}

# Calculate average scores for each subject
# Since the structure of the dictionary is consistent for all students, we can extract the list of subjects by accessing any student's dictionary.
for subject in student_scores['Doma']: 
    total_score = sum(student_scores[student][subject] for student in student_scores)
    average_scores[subject] = total_score / len(student_scores)

colors = ['blue', 'green', 'red']

# Plot a bar chart
plt.bar(average_scores.keys(), average_scores.values(), color=colors)

# Adding labels and title
plt.xlabel('Subjects')
plt.ylabel('Average Scores')
plt.title('Average Scores of each student in Different Subjects')
plt.show()
