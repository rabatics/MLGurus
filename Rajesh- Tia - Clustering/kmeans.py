from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv
import numpy as np

file_name = "result.csv"


class Student():
	def __init__(self):
		self.Anon_Student_Id = None
		self.CFA_Percentage = None
		self.Correct_Percentage = None
		self.label = None

student_list = []

with open(file_name) as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		student = Student()
		student.Anon_Student_Id = row['Anon Student Id']
		student.CFA_Percentage = row['CFA_Percentage']
		student.Correct_Percentage = row['Correct_Percentage']
		student_list.append(student)

Matrix = np.zeros((len(student_list), 5))

i = 0
for ele in student_list:
	Matrix[i][0] = ele.CFA_Percentage
	Matrix[i][1] = ele.Correct_Percentage
	i += 1


estimator = KMeans(n_clusters=7, n_init=1,init='random')
estimator.fit(Matrix)
labels = estimator.labels_

i = 0
fig = plt.figure()
for ele in student_list:
	ele.label = labels[i]
	i += 1
	# if ele.label == 6:
	# 	color = 'm'
	# if ele.label == 5:
	# 	color = 'k'
	# if ele.label == 4:
	# 	color = 'b'
	# if ele.label == 3:
	# 	color = 'c'
	# if ele.label == 2:
	# 	color = 'r'
	# if ele.label == 1:
	# 	color = 'g'
	# if ele.label  == 0:
	# 	color = 'y'
	plt.scatter(ele.CFA_Percentage, ele.Correct_Percentage, c = 'b')
fig.suptitle('K Means, k = 7')
plt.xlabel('CFA_Percentage')
plt.ylabel('Correct_Percentage')
plt.show()


