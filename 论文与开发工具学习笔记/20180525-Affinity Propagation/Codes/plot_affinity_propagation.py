"""
=================================================
Demo of affinity propagation clustering algorithm
=================================================

Reference:
Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
Between Data Points", Science Feb. 2007

"""
print(__doc__)

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='dark', font_scale=1.2, palette='deep', color_codes=True)
np.random.seed(0)
# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=311,
                            centers=centers,
                            cluster_std=0.5,
                            random_state=0)

# #############################################################################

# Compute Affinity Propagation
ap = AffinityPropagation(preference=-2,
                         damping=0.6,
                         max_iter=500,
                         convergence_iter=10)
ap.fit(X)
cluster_centers_indices = ap.cluster_centers_indices_
labels = ap.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure()
plt.scatter(X[:,0], X[:,1], color='b', marker='.')
plt.savefig('dist.png', dpi=600, bbox_inches='tight')

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.savefig("..//Plots//ClusterResults.pdf", bbox_inches='tight')

# #############################################################################
# Cluster numbers with the changing of the preference
'''
preferenceList = list(range(-200, 0, 10))
n_clusters = []
for p in preferenceList:
    ap = AffinityPropagation(preference=p,
                         damping=0.6,
                         max_iter=300,
                         convergence_iter=20)
    ap.fit(X)
    n_clusters.append(len(ap.cluster_centers_indices_))

f, ax = plt.subplots(figsize = (14, 10))
plt.title("Relationship between preference and number of clusters")
plt.plot(preferenceList, n_clusters, 'k-s')
plt.xlabel("Preference")
plt.ylabel("Cluster numbers")
plt.grid(True)
plt.savefig("..//Plots//PreferenceAndNumberOfClusters.pdf", dpi=800, bbox_inches='tight')


dampingList = np.arange(0.5, 1, 0.05)
iterList = []
for i in dampingList:
    ap = AffinityPropagation(preference=-50,
                         damping=i,
                         max_iter=500,
                         convergence_iter=20)
    ap.fit(X)
    iterList.append(ap.n_iter_)
iterList = np.array(iterList)
f, ax = plt.subplots(figsize = (14, 10))
plt.title("Damping value and n_iters with perference=-50")
plt.plot(dampingList, iterList, 'k-s')
plt.xlabel("damping")
plt.ylabel("n_iters")
plt.grid(True)
plt.savefig("..//Plots//itersAndDamping.pdf", dpi=800, bbox_inches='tight')
'''
