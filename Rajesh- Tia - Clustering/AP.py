from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import csv
import numpy as np

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
  #                          random_state=0)

  # read_data_from_file

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

Matrix = np.zeros((len(student_list), 2))

i = 0
for ele in student_list:
  Matrix[i][0] = ele.CFA_Percentage
  Matrix[i][1] = ele.Correct_Percentage
  i += 1

# Matrix,labels_true
    

##############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(preference=-0.25,affinity="euclidean").fit(Matrix)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
print (labels)

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
      #print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels))
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

##############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = Matrix[cluster_centers_indices[k]]
    plt.plot(Matrix[class_members, 0], Matrix[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in Matrix[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

