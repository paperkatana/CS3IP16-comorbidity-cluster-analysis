import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import random
from collections import Counter
from scipy.sparse import hstack
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df = pd.read_csv("DATA.csv")
print("data imported")
df = df.drop('Unnamed: 0', axis=1)
print(df.head())

# create a feature matrix with primary and secondary diagnoses as features
X = df[['PRIMARY_DIAGNOSIS', 'SECONDARY_DIAGNOSIS']].to_numpy()
print(X.shape)


#perform principal component analysis
pca = PCA(n_components=2)
print("initialised pca")
XPCA = pca.fit_transform(X)
print("fitted pca to multi label binarized data")
print(XPCA.shape)

"""
tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000)
print("created tsne object")
xtsne = tsne.fit_transform(X)
print("tsne object fit to xpca data")
"""
#K = [4]
K = [4, 6, 8, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500, 600]

#create a custom colormap for up to 600 unique colours
cmap = ListedColormap(np.random.rand(600,3))

silhouetteScores1 = []
print("silhouette score list initialised")
chScores1 = []
print("calinski harabasz index list initialised")
sseScores1 = []
print("sse score list initialised")

for k in K:
    print(f'k = {k}')

    # perform K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(XPCA)
    print("kmeans fit to k")
    """
    #silhouette score
    #aiming for as close to 1.0 as possible
    silhouette = silhouette_score(XPCA, kmeans.labels_)
    silhouetteScores1.append(silhouette)
    print("The average silhouette_score is :", silhouette)
    """
    #calinski-harabasz index (also known as the Variance Ratio Criterion)
    #aiming for a higher value
    chIndex = calinski_harabasz_score(XPCA, kmeans.labels_)
    chScores1.append(chIndex)
    print('Calinski-Harabasz Index:', chIndex)

    #sse
    #aiming for a lower value
    #The inertia_ attribute returns the sum of squared distances of all data points to their closest centroid
    sse = kmeans.inertia_
    sseScores1.append(sse)
    print('SSE:', sse)

    mask0 = df['EXPIRE_FLAG'] == 0
    mask1 = df['EXPIRE_FLAG'] == 1
    
    X0 = X[mask0]
    X1 = X[mask1]
    
    labels0 = kmeans.labels_[mask0]
    labels1 = kmeans.labels_[mask1]

    plt.figure(figsize=(15,15))
    plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
    plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

    plt.xlabel('Primary diagnosis')
    plt.ylabel('Secondary diagnosis')
    plt.title(str(k) + ' Clusters')

    filename = './kmeans/' + str(k) + '_clusters.png'
    plt.savefig(filename)
    plt.show()
    

    """
    plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap=cmap)
    filename = './kmeans/' + str(k) + '_clusters.png'
    plt.savefig(filename)
    plt.show()
    
    for i in range(len(X)):
        if df.loc[i, 'EXPIRE_FLAG'] == 0:
            plt.scatter(X[i,0], X[i,1], c=kmeans.labels_[i], marker='o', cmap='rainbow')
        else:
            plt.scatter(X[i,0], X[i,1], c=kmeans.labels_[i], marker='x', cmap='rainbow')

    filename = './kmeans/' + str(k) + '_clusters.png'
    plt.savefig(filename)
    plt.show()
    """
