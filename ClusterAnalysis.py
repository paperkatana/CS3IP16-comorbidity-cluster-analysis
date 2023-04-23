import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import random
import re
from scipy.sparse import hstack
from scipy.spatial.distance import cdist, euclidean
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from sklearn.neighbors import NearestNeighbors

print("all packages imported")
#---------------------------------------------------------------------------------------------
#Data cleaning

#import the patients CSV file into a DataFrame and check the contents
patients = pd.read_csv('./mimic-iii/PATIENTS.csv')
#print(len(patients.index))
#print(patients.head())

#import the diagnoses CSV file into a DataFrame and check the contents
diagnoses = pd.read_csv('./mimic-iii/DIAGNOSES_ICD.csv')
#print(len(diagnoses.index))
#print(diagnoses.head())

#remove all ICD 9 codes that aren't exclusively numeric
diagnoses['ICD9_CODE'] = diagnoses['ICD9_CODE'].astype(str)
diagnoses = diagnoses[diagnoses['ICD9_CODE'].apply(lambda x: re.match(r'^\d+$', x) is not None)]
#print(len(diagnoses.index))
#print(diagnoses.head())

# filter out admission_ids without a primary diagnosis
diagnoses = diagnoses.groupby('HADM_ID').filter(lambda x: (x['SEQ_NUM'] == 1.0).any())
#print(len(diagnoses.index))
#print(diagnoses.head())

#split into primary/secondary diagnoses
diagnosesPrimary = diagnoses[diagnoses['SEQ_NUM'] == 1.0].copy()
diagnosesSec = diagnoses[diagnoses['SEQ_NUM'] != 1.0].copy()
#print(diagnosesPrimary.head())
#print(diagnosesSec.head())

diagnosesPrimary.drop(columns=['ROW_ID', 'SEQ_NUM'], inplace=True)
#remove duplicate rows
diagnosesPrimary.drop_duplicates(inplace=True)
#group by admission ID
diagnosesPrimary['ICD9_CODE'] = diagnosesPrimary.groupby(['HADM_ID'])['ICD9_CODE'].transform(lambda x: ','.join(x))

#repeat for secondary diagnoses
diagnosesSec.drop(columns=['ROW_ID', 'SEQ_NUM'], inplace=True)
diagnosesSec.drop_duplicates(inplace=True)
diagnosesSec['ICD9_CODE'] = diagnosesSec.groupby(['HADM_ID'])['ICD9_CODE'].transform(lambda x: ','.join(x))

diagnosesPrimary.drop_duplicates()
diagnosesSec.drop_duplicates()

patients.drop(columns=['ROW_ID', 'GENDER', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN'], inplace=True)
#print(patients.head())

outputData = pd.merge(patients, diagnosesPrimary, on='SUBJECT_ID')
outputData.drop_duplicates(inplace=True)

outputData = pd.merge(outputData, diagnosesSec, on='HADM_ID')
outputData.drop_duplicates(inplace=True)

outputData.rename(columns={'ICD9_CODE_x':'PRIMARY_DIAGNOSIS', 'ICD9_CODE_y':'SECONDARY_DIAGNOSES', 'SUBJECT_ID_x':'SUBJECT_ID'}, inplace=True)
expireFlag = outputData.pop('EXPIRE_FLAG')
outputData.insert(len(outputData.columns), 'EXPIRE_FLAG', expireFlag)
outputData.drop(columns=['SUBJECT_ID_y'], inplace=True)
outputData.drop_duplicates(inplace=True)
outputData['PRIMARY_DIAGNOSIS'] = outputData['PRIMARY_DIAGNOSIS'].astype(int)
print(outputData.head())

outputData.to_csv('./data/ADMISSIONS_PRIMARY_SECONDARY.csv')

diag = outputData.set_index(['SUBJECT_ID', 'HADM_ID', 'PRIMARY_DIAGNOSIS', 'EXPIRE_FLAG'])['SECONDARY_DIAGNOSES']\
            .str.split(',', expand=True)\
            .stack()\
            .reset_index(name='SECONDARY_DIAGNOSIS')\
            .drop('level_4', axis=1)

# merge the original dataframe with the resulting dataframe to get the final output
diagData = outputData.merge(diag, on=['SUBJECT_ID', 'HADM_ID', 'PRIMARY_DIAGNOSIS', 'EXPIRE_FLAG'])
diagData = diagData.drop('SECONDARY_DIAGNOSES', axis=1)
diagData['SECONDARY_DIAGNOSIS'] = diagData['SECONDARY_DIAGNOSIS'].astype(int)
diagData.drop_duplicates()
# print the resulting dataframe
#print(diagData.head())
diagData.to_csv('./data/DATA.csv')

#---------------------------------------------------------------------------------------------
#Cluster analysis preparation


# create a feature matrix with primary and secondary diagnoses as features
X = diagData[['PRIMARY_DIAGNOSIS', 'SECONDARY_DIAGNOSIS']].to_numpy()
print(X.shape)

#count how many instances of each ICD9 code in the primary diagnosis column
primCount = outputData['PRIMARY_DIAGNOSIS'].value_counts(normalize=False)
#print("Primary diagnoses counted")

#count how many instances of each ICD9 code in the secondary diagnosis column
secCount = diagData['SECONDARY_DIAGNOSIS'].value_counts(normalize=False)
#print("Secondary diagnoses counted")

#perform principal component analysis
pca = PCA(n_components=2)
#print("initialised pca")
XPCA = pca.fit_transform(X)
#print("fitted pca to multi label binarized data")
#print(XPCA.shape)

X = diagData[['PRIMARY_DIAGNOSIS', 'SECONDARY_DIAGNOSIS', 'EXPIRE_FLAG']].to_numpy()

"""
tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000)
print("created tsne object")
xtsne = tsne.fit_transform(X)
print("tsne object fit to xpca data")
"""

#K = [4]
K = [10, 40, 80, 100, 150, 200, 250, 300, 400, 500, 600]
batchSize = [50, 100, 200, 500, 1000]
print("values of k set")

#create a custom colormap for up to 600 unique colours
cmap = ListedColormap(np.random.rand(600,3))

mask0 = diagData['EXPIRE_FLAG'] == 0
mask1 = diagData['EXPIRE_FLAG'] == 1
    
X0 = X[mask0]
X1 = X[mask1]
#---------------------------------------------------------------------------------------------
"""
#K-Means

silhouetteScores1 = []
print("silhouette score list initialised")
chScores1 = []
print("calinski harabasz index list initialised")
dbiScores1 = []
print("davies-bouldin index list initialised")
sseScores1 = []
print("sse score list initialised")

for k in K:
    print(f'k = {k}')

    # perform K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(XPCA)
    print("kmeans fit to k")

    
    #silhouette score
    #aiming for as close to 1.0 as possible
    #silhouette = silhouette_score(X, kmeans.labels_)
    #silhouetteScores1.append(silhouette)
    #print("The average silhouette_score is :", silhouette)
    

    #calinski-harabasz index (also known as the Variance Ratio Criterion)
    #aiming for a higher value
    chIndex = calinski_harabasz_score(XPCA, kmeans.labels_)
    chScores1.append(chIndex)
    print('Calinski-Harabasz Index:', chIndex)
    print(chScores1)

    #davies-bouldin index 
    #aiming for a lower value
    dbiIndex = davies_bouldin_score(XPCA, kmeans.labels_)
    dbiScores1.append(dbiIndex)
    print('Davies-Bouldin Index:', dbiIndex)
    print(dbiScores1)

    #sse
    #aiming for a lower value
    #The inertia_ attribute returns the sum of squared distances of all data points to their closest centroid
    sse = kmeans.inertia_
    sseScores1.append(sse)
    print('SSE:', sse)
    print(sseScores1)
    
    labels0 = kmeans.labels_[mask0]
    labels1 = kmeans.labels_[mask1]

    fig, ax = plt.subplots(ncols=2, figsize=(20,10))

    ax[0].scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
    ax[0].scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)
    ax[1].scatter(X[:, 0], X[:, 1], c=X[:, 2], s=10, cmap=cmap)

    ax[0].set_xlabel('Primary diagnosis')
    ax[0].set_ylabel('Secondary diagnosis')
    ax[0].set_title(str(k) + ' Clusters (k-Means)')
    ax[1].set_xlabel('Primary diagnosis')
    ax[1].set_ylabel('Secondary diagnosis')
    ax[1].set_title('Discharged vs deceased patients')

    filename = './kmeans/kmeans_' + str(k) + '_clusters.png'
    plt.savefig(filename)
    #plt.show()

fig, ax = plt.subplots(nrows=3, figsize=(5,15))

ax[0].plot(K, chScores1)
ax[1].plot(K, dbiScores1)
ax[2].plot(K, sseScores1)

ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('Calinki-Harabasz Index')
ax[0].set_title('Calinski-Harabasz Scores')
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Davies-Bouldin Index')
ax[1].set_title('Davies-Bouldin Scores')
ax[2].set_xlabel('Number of clusters')
ax[2].set_ylabel('Sum of Squared Error Score')
ax[2].set_title('SSE Scores')

plt.tight_layout()

filename = './kmeans/kmeans_clusters_metrics.png'
plt.savefig(filename)
#plt.show()
 
#---------------------------------------------------------------------------------------------

#mini batch kmeans

silhouetteScores2 = []
print("silhouette score list initialised")
chScores2 = []
print("calinski harabasz index list initialised")
dbiScores2 = []
print("davies-bouldin index list initialised")
sseScores2 = []
print("sse score list initialised")

for k in K:
    for b in batchSize:
        print(f'k = {k}. batch size = {b}')

        chScoresTemp = []
        dbiScoresTemp = []
        sseScoresTemp = []

        # perform clustering
        mbk = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=b, random_state=0)
        mbk.fit(XPCA)
        #what does different batch size do?

        #silhouette score
        #aiming for as close to 1.0 as possible
        #silhouette = silhouette_score(XPCA, clusterLabels)
        #silhouetteScores2.append(silhouette)
        #print("The average silhouette_score is :", silhouette)

        #calinski-harabasz index (also known as the Variance Ratio Criterion)
        #aiming for a higher value
        chIndex = calinski_harabasz_score(XPCA, mbk.labels_)
        chScoresTemp.append(chIndex)
        chScores2.append(chIndex)
        print('Calinski-Harabasz Index:', chIndex)

        #davies-bouldin index 
        #aiming for a lower value
        dbiIndex = davies_bouldin_score(XPCA, mbk.labels_)
        dbiScoresTemp.append(bbiIndex)
        dbiScores2.append(dbiIndex)
        print('Davies-Bouldin Index:', dbiIndex)

        #sse
        #aiming for a lower value
        #The inertia_ attribute returns the sum of squared distances of all data points to their closest centroid
        sse = mbk.inertia_
        sseScoresTemp.append(sse)
        sseScores2.append(sse)
        print('SSE:', sse)

        labels0 = mbk.labels_[mask0]
        labels1 = mbk.labels_[mask1]

        fig, ax = plt.subplots(ncols=2, figsize=(20,10))

        ax[0].scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
        ax[0].scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)
        ax[1].scatter(X[:, 0], X[:, 1], c=X[:, 2], s=10, cmap=cmap)

        ax[0].set_xlabel('Primary diagnosis')
        ax[0].set_ylabel('Secondary diagnosis')
        ax[0].set_title(str(k) + ' Clusters (Mini Batch k-Means, batch size ' + str(b) + ')')
        ax[1].set_xlabel('Primary diagnosis')
        ax[1].set_ylabel('Secondary diagnosis')
        ax[1].set_title('Discharged vs deceased patients')

        filename = './mbk/mbk_' + str(k) + '_clusters_' + str(b) + 'batchsize.png'
        plt.savefig(filename)
        #plt.show()

    fig, ax = plt.subplots(ncols=3, figsize=(25,15))

    ax[0].plot(K, chScoresTemp)
    ax[1].plot(K, dbiscoresTemp)
    ax[2].plot(K, sseScoresTemp)

    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('Calinki-Harabasz Index')
    ax[0].set_title('Calinski-Harabasz Scores')
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('Davies-Bouldin Index')
    ax[1].set_title('Davies-Bouldin Scores')
    ax[2].set_xlabel('Number of clusters')
    ax[2].set_ylabel('Sum of Squared Error Score')
    ax[2].set_title('SSE Scores')

    filename = './mbk/mbk_' + str(k) + '_clusters_metrics.png'
    plt.savefig(filename)
    #plt.show()
"""  
#---------------------------------------------------------------------------------------------
"""
#agglomerative

silhouetteScores2 = []
print("silhouette score list initialised")
chScores2 = []
print("calinski harabasz index list initialised")
dbiScores2 = []
print("davies-bouldin index list initialised")
sseScores2 = []
print("sse score list initialised")

for k in K:
    print(f'k = {k}')
    
    nn = NearestNeighbors(n_neighbors=10) #vary n_neighbors?
    nn.fit(XPCA)
    distances, indices = nn.kneighbors(XPCA)
    eps = np.mean(distances[:, 1:])
    minSamples = int(0.05 * XPCA.shape[0]) #vary 0.05?
    #epsRange = np.linspace(0.5 * avgDist, 1.5 * avgDist, num=10)
    #minSamplesRange = np.linspace(0.02 * XPCA.shape[0], 0.1 * XPCA.shape[0], num=10)

    # perform K-means clustering
    model = DBSCAN(eps=eps, min_samples=minSamples)
    model.fit(XPCA)
    print("dbscan fit to k")

    #silhouette score
    #aiming for as close to 1.0 as possible
    #silhouette = silhouette_score(XPCA, clusterLabels)
    #silhouetteScores2.append(silhouette)
    #print("The average silhouette_score is :", silhouette)

    #calinski-harabasz index (also known as the Variance Ratio Criterion)
    #aiming for a higher value
    chIndex = calinski_harabasz_score(XPCA, model.labels_)
    chScores2.append(chIndex)
    print('Calinski-Harabasz Index:', chIndex)

    #davies-bouldin index 
    #aiming for a lower value
    dbiIndex = davies_bouldin_score(XPCA, model.labels_)
    dbiScores2.append(dbiIndex)
    print('Davies-Bouldin Index:', dbiIndex)

    #sse
    #aiming for a lower value
    #The inertia_ attribute returns the sum of squared distances of all data points to their closest centroid
    centers = np.array([XPCA[labels == i].mean(axis=0) for i in range(k)])
    distances = np.array([np.linalg.norm(XPCA[labels == i] - centers[i], axis=1) for i in range(k)])
    sse = np.sum(distances ** 2)
    sseScores2.append(sse)
    print('SSE:', sse)

    labels0 = model.labels_[mask0]
    labels1 = model.labels_[mask1]

    fig, ax = plt.subplots(ncols=2, figsize=(20,10))

    ax[0].scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
    ax[0].scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)
    ax[1].scatter(X[:, 0], X[:, 1], c=X[:, 2], s=10, cmap=cmap)

    ax[0].set_xlabel('Primary diagnosis')
    ax[0].set_ylabel('Secondary diagnosis')
    ax[0].set_title(str(k) + ' Clusters (DBSCAN)')
    ax[1].set_xlabel('Primary diagnosis')
    ax[1].set_ylabel('Secondary diagnosis')
    ax[1].set_title('Discharged vs deceased patients')

    filename = './dbscan/dbscan_' + str(k) + '_clusters.png'
    plt.savefig(filename)
    plt.show()

fig, ax = plt.subplots(ncols=3, figsize=(25,15))

ax[0].plot(K, chScores2)
ax[1].plot(K, dbiscores2)
ax[2].plot(K, sseScores2)

ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('Calinki-Harabasz Index')
ax[0].set_title('Calinski-Harabasz Scores')
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Davies-Bouldin Index')
ax[1].set_title('Davies-Bouldin Scores')
ax[2].set_xlabel('Number of clusters')
ax[2].set_ylabel('Sum of Squared Error Score')
ax[2].set_title('SSE Scores')

filename = './dbscan/dbscan_clusters_metrics.png'
plt.savefig(filename)
plt.show()

"""


#---------------------------------------------------------------------------------------------
"""
#CLARA

silhouetteScores3 = []
print("silhouette score list initialised")
chScores3 = []
print("calinski harabasz index list initialised")
dbiScores3 = []
print("davies-bouldin index list initialised")
sseScores3 = []
print("sse score list initialised")

maxNeighbours = [2,5,8,10,12]
numLocal = [3,4,5]

for k in K:
    for n in maxNeighbours:
        print(f'k = {k}. maxNeighbours = {n}')

        chScoresTemp = []
        sseScoresTemp = []

        # perform clustering
        clara = clarans(XPCA, k, n, 3)
        print("CLARA created")
        clara.process()
        print("CLARA processed")
        clusters = clara.get_clusters()
        medoids = clara.get_medoids()

        #silhouette score
        #aiming for as close to 1.0 as possible
        #silhouette = silhouette_score(XPCA, clusterLabels)
        #silhouetteScores2.append(silhouette)
        #print("The average silhouette_score is :", silhouette)

        #calinski-harabasz index (also known as the Variance Ratio Criterion)
        #aiming for a higher value
        chIndex = calinski_harabasz_score(XPCA, clusters)
        chScoresTemp.append(chIndex)
        chScores3.append(chIndex)
        print('Calinski-Harabasz Index:', chIndex)

        #davies-bouldin index 
        #aiming for a lower value
        dbiIndex = davies_bouldin_score(XPCA, clusters)
        dbiScores1.append(dbiIndex)
        print('Davies-Bouldin Index:', dbiIndex)

        #sse
        #aiming for a lower value
        #The inertia_ attribute returns the sum of squared distances of all data points to their closest centroid
        distances = cdist(XPCA, medoids, 'euclidean')
        sse = np.min(distances, axis=1).sum()
        sseScoresTemp.append(sse)
        sseScores3.append(sse)
        print('SSE:', sse)

        labels0 = clusters[mask0]
        labels1 = clusters[mask1]

        fig, ax = plt.subplots(ncols=2, figsize=(20,10))

        ax[0].scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
        ax[0].scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)
        ax[1].scatter(X[:, 0], X[:, 1], c=X[:, 2], s=10, cmap=cmap)

        ax[0].set_xlabel('Primary diagnosis')
        ax[0].set_ylabel('Secondary diagnosis')
        ax[0].set_title(str(k) + ' Clusters (CLARA, maxNeighbor=' + str(n) + ')')
        ax[1].set_xlabel('Primary diagnosis')
        ax[1].set_ylabel('Secondary diagnosis')
        ax[1].set_title('Discharged vs deceased patients')

        filename = './clara/clara_' + str(k) + '_clusters_' + str(n) + 'maxneighbor.png'
        plt.savefig(filename)
        #plt.show()

    fig, ax = plt.subplots(ncols=3, figsize=(25,15))

    ax[0].plot(K, chScores3)
    ax[1].plot(K, dbiscores3)
    ax[2].plot(K, sseScores3)

    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('Calinki-Harabasz Index')
    ax[0].set_title('Calinski-Harabasz Scores')
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('Davies-Bouldin Index')
    ax[1].set_title('Davies-Bouldin Scores')
    ax[2].set_xlabel('Number of clusters')
    ax[2].set_ylabel('Sum of Squared Error Score')
    ax[2].set_title('SSE Scores')

    filename = './kmeans/kmeans_' + str(k) + '_clusters_metrics.png'
    plt.savefig(filename)
    #plt.show()

plt.show()
"""
#---------------------------------------------------------------------------------------------

#K-Algorithm

def RelativeRisk(codeA, codeB, eventFlag):
    #print("Beginning RelativeRisk(codeA, codeB)")
    #print(f'Primary diagnosis: {codeA}. Secondary diagnosis: {codeB}.')
    countTotal = primCount.sum() + secCount.sum()
    #print(f'Count total: {countTotal}. primCount={primCount.sum()} and secCount={secCount.sum()}')
    countA = primCount.loc[codeA] if codeA in primCount.index else 0
    #print(f'Count of primary diagnosis {codeA}: {countA}')
    countB = secCount.loc[codeB] if codeB in secCount.index else 0
    #print(f'Count of secondary diagnosis {codeB}: {countB}')
    countAB = countA + countB
    #print(f'count A + count B = {countAB}')
    relativeRisk = (countAB / countTotal) / ((countA / countTotal)*(countB / countTotal))
    if eventFlag == 1:
        relativeRisk *= 2
    #print(f'Relative risk: {relativeRisk}')
    #print("Ending RelativeRisk(codeA, codeB)")
    return relativeRisk

def ClusterRRTotal(pClusterSet, pClusterIndex):
    #print("Beginning ClusterRRTotal(clusterIndex)")
    #print(f'pClusterIndex: {pClusterIndex}')
    # Get the indices of the data points in the current cluster
    indices = np.where(pClusterSet.labels_ == pClusterIndex)[0]
    #print(indices)
    #print(f'all indices for clusterIndex {pClusterIndex} retrieved')
    clusterTotal = 0.0
    #print("clusterTotal initialised")
    # Loop through all possible pairs of data points in the cluster
    for i in range(len(indices)):
        # Calculate the relative risk for the pair of data points
        primDiag = X[indices[i], 0]
        secDiag = X[indices[i], 1]
        eventFlag = X[indices[i], 2]
        relativeRisk = RelativeRisk(primDiag, secDiag, eventFlag)
        clusterTotal += relativeRisk
    #print("Looped through all possible pairs of data points in the cluster and calculated relative risk")
    #print(f'Final cluster total for cluster {pClusterIndex}: {clusterTotal}')
    #print("Ending ClusterRRTotal(clusterIndex)")
    return clusterTotal

def KAlgorithm(pClusterSet, k):
    labels = pClusterSet.labels_
    indices = np.arange(len(XPCA))
    np.random.shuffle(indices)
    changeMade = False
    threshold = 0
    
    while True:
        #process every data point in a random order
        for i in indices:
            oldCluster = pClusterSet.predict(XPCA[i].reshape(1,-1))[0]
            newCluster = oldCluster
            bestDelta = 0
            updatedKMeans = pClusterSet

            #loop all clusters
            for cluster in range(k):
                #calculate this cluster's current rr total
                currentRR = ClusterRRTotal(pClusterSet, cluster)
                print(f'CurrentRR: {currentRR}')
                #calculate the new rr total if i moved to this cluster
                tempLabels = pClusterSet.labels_.copy()
                tempLabels[i] = cluster
                indicesOld = np.where(tempLabels == oldCluster)[0]
                indicesNew = np.where(tempLabels == cluster)[0]

                clusterOldUpdated = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA[indicesOld])
                tempClusterSet = np.concatenate((pClusterSet.cluster_centers_, clusterOldUpdated.cluster_centers_), axis=0)
                tempKMeans1 = KMeans(n_clusters=k, n_init=10, random_state=0)
                tempKMeans1.cluster_centers_ = tempClusterSet

                clusterNewUpdated = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA[indicesNew])
                tempClusterSet = np.concatenate((tempKMeans1.cluster_centers_, clusterNewUpdated.cluster_centers_), axis=0)
                newKMeans2 = KMeans(n_clusters=k, n_init=10, random_state=0)
                newKMeans2.cluster_centers_ = tempClusterSet
                newKMeans2.labels_ = tempLabels

                newRR = ClusterRRTotal(newKMeans2, cluster)
                print(f'newRR: {newRR}')
                delta = newRR - currentRR
                
                if delta > bestDelta:
                    print(f'Delta {delta} > bestDelta {bestDelta}')
                    bestDelta = delta
                    newCluster = cluster
                    updatedKMeans = newKMeans2
                else:
                    print(f'Delta {delta} < bestDelta {bestDelta}')

            if bestDelta > threshold:
                print(f'Best Delta: {bestDelta}. Greater than {threshold}')
                #move the actual datapoint
                pClusterSet = updatedKMeans
                changeMade = True

        if changeMade == False:
            break


"""
clusterSet = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA)
print("Cluster set initialised with one cluster")

silhouetteScores4 = []
print("silhouette score list initialised")
chScores4 = []
print("calinski harabasz index list initialised")
dbiScores4 = []
print("davies-bouldin index list initialised")
sseScores4 = []
print("sse score list initialised")

for k in K:
    clusterSet = KMeans(n_clusters=k, n_init=10, random_state=0).fit(XPCA)
    print(f'Cluster set for k={k} created')
    KAlgorithm(clusterSet, k)
    print("K Algorithm successfully performed")

    #silhouette = silhouette_score(XPCA, clusterSet.labels_)
    #print(f'Silhouette Average: {silhouette}')
    chIndex = calinski_harabasz_score(XPCA, clusterSet.labels_)
    print(f'Calinski Harabasz Score: {chIndex}')
    dbiIndex = davies_bouldin_score(XPCA, clusterSet.labels_)
    print('Davies-Bouldin Index:', dbiIndex)
    sse = clusterSet.inertia_
    print(f'SSE Score: {sse}')
    
    labels0 = clusterSet.labels_[mask0]
    labels1 = clusterSet.labels_[mask1]

    plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
    plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

    plt.xlabel('Primary diagnosis')
    plt.ylabel('Secondary diagnosis')
    plt.title(str(k) + ' Clusters (K-Algorithm)')

    filename = './kalg/k_algorithm_' + str(k) + '_clusters.png'
    plt.savefig(filename)
    plt.show()
"""
#---------------------------------------------------------------------------------------------

#M-Algorithm

"""
assign the current cluster solution to a new variable to be changed
#calculate the weights between the nodes(?)
#randomly select two clusters where the sum of their weights divided by the sum of all weights is greater than a threshold
#merge the two clusters
#select another random cluster from the list and a random unbalance factor between 5-95%
#grow a new cluster i within this cluster by selecting a random seed within this cluster
#then search for the node with the highest sum of edge weights and merge it with i
#repeat until percentage is reached or there are no more nodes where the sum of edge weights is greater than 0
#after splitting the cluster, perform k-means
#calculate the cost function for this new cluster set and if it is better than the previous one, update cluster set
#otherwise, discard it
#repeat this r times
"""

def calcE(pClusterSet, k):
    tempE = 0.0
    # Loop through each cluster
    for i in range(k):
        clusterTotal = ClusterRRTotal(pClusterSet, i)
        tempE += clusterTotal
    print(f'Total of relative risks for each cluster: {tempE}')
    return tempE

def calcSSE(pClusterSet):
    centers = pClusterSet.cluster_centers_
    labels = pClusterSet.labels_
    sse = 0

    for i in range(len(XPCA)):
        center = centers[labels[i]]
        dist = euclidean(XPCA[i], center)
        squaredDist = dist**2
        sse += squaredDist

    return sse
            
def GrowCluster(pClusterSet):
    #print("Beginning GrowCluster(pClusterSet, pClusterC, pUnbalanceFactor)")
    #print(f'pClusterC: {pClusterC}. pUnbalanceFactor: {pUnbalanceFactor}')
    #get clusterC data
    print(pClusterSet.labels_)
    labels = pClusterSet.labels_.copy()
    centers = pClusterSet.cluster_centers_
    print(len(centers))

    ssePerCluster = np.zeros(pClusterSet.n_clusters)
    for i in range(len(XPCA)):
        center = centers[labels[i]]
        dist = euclidean(XPCA[i], center)
        squaredDist = dist**2
        ssePerCluster[labels[i]] += squaredDist

    #clusterC = random.randint(0, pClusterSet.n_clusters)
    clusterC = np.argmax(ssePerCluster)
    print(f'Cluster {clusterC} chosen')
    indicesC = np.where(labels == clusterC)[0]
    print(indicesC)
    #print("created list of data points for cluster C")
    
    #find out how many nodes should be in the new cluster (% of original cluster)
    unbalanceFactor = random.uniform(0.05, 0.95)
    print(f'Unbalance Factor {unbalanceFactor} selected')
    targetSize = int(len(indicesC) * unbalanceFactor)
    print(f'Calculated target size: {targetSize}')
    
    """
    randomPrimIndex = np.random.choice(indicesC)
    indicesD = [randomPrimIndex]
    indicesC = np.delete(indicesC, np.where(indicesC == randomPrimIndex))
    #print("Removed primary diagnosis from cluster C")

    relativeRisks = []
    for index in indicesC:
        relativeRisks.append(RelativeRisk(X[randomPrimIndex, 0], X[index, 1], X[index, 2]))
    #print("Created list of relative risks for every data point in cluster C, in comparison to the primary diagnosis in cluster D")
        
    sortedIndicesC = [x for _, x in sorted(zip(relativeRisks, indicesC), reverse=True)]
    indicesD = sortedIndicesC[:targetSize]
    indicesC = sortedIndicesC[targetSize:]
    print(f'IndicesD: {len(indicesD)}. Target Size: {targetSize}. IndicesC: {len(indicesC)}')
    """
    
    newClusterC = KMeans(n_clusters=2, n_init=10, random_state=0).fit(XPCA[indicesC])
    pClusterSetCenters = np.concatenate((pClusterSet.cluster_centers_, newClusterC.cluster_centers_), axis=0)
    tempKMeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    tempKMeans.cluster_centers_ = pClusterSetCenters
    
    indicesC = np.where(newClusterC == 0)[0]
    indicesD = np.where(newClusterC == 1)[0]

    labels[indicesC] = k-1
    labels[indicesD] = k

    pClusterSet.labels_ = labels
    print(f'Final labels: {pClusterSet.labels_}')
    #print(f'Inertia: {newKMeans.inertia_}')
    
    return pClusterSet


#def MergeClusters(pClusterSet, pE, k):

def MAlgorithm(pClusterSet, k):
    print(f'Beginning MAlgorithm(k). k = {k}')
    #create copy of prevaling cluster set as newClusterSet
    newClusterSet = pClusterSet
    begE = calcE(newClusterSet, k)
    begSSE = calcSSE(newClusterSet)
    print(begSSE)

    splitClusterSet = GrowCluster(newClusterSet)
    print("New cluster grown")

    tempE = calcE(splitClusterSet, k)
    print(f'E after merging: {tempE}')
    tempSSE = calcSSE(splitClusterSet)
    print(f'Merged clusters inertia: {tempSSE}')

    clusterA = 0
    clusterB = 0
    threshold = 0.25    #or 1/k *2?
    print(f'Threshold set to {threshold}')
    thresholdMet = False
    while thresholdMet == False:
        clusterA = random.randint(0, splitClusterSet.n_clusters)
        print(f'Picked random cluster A: {clusterA}')
        clusterB = random.randint(0, splitClusterSet.n_clusters)
        print(f'Picked random cluster B: {clusterB}')
        if clusterA == clusterB:
            clusterB += 1
            print(f'clusters A and B are the same. New cluster B: {clusterB}')
        #calculate sum of their weights
        clusterARR = ClusterRRTotal(splitClusterSet, clusterA)
        #print(f'Cluster ARR: {clusterARR}')
        clusterBRR = ClusterRRTotal(splitClusterSet, clusterB)
        #print(f'Cluster BRR: {clusterBRR}')
        probability = (clusterARR + clusterBRR) / begE
        print(f'probability = {clusterARR} + {clusterBRR} / {begE} = {probability}')
        if probability > threshold:
            print("Probability is greater than threshold")
            thresholdMet = True

    newLabels = splitClusterSet.labels_.copy()
    newLabels = np.where(newLabels == clusterA, 999, newLabels)
    newLabels = np.where(newLabels == clusterB, 999, newLabels)

    # delete the original clusters from the labels array
    newLabels = np.delete(newLabels, np.where(newLabels == clusterA))
    newLabels = np.delete(newLabels, np.where(newLabels == clusterB))
    #print("Deleted clusters A and B from the labels")

    uniqueLabels = np.unique(newLabels[newLabels != 999])
    labelMap = {label: i for i, label in enumerate(uniqueLabels)}
    newLabels = np.array([labelMap[label] if label != 999 else -1 for label in newLabels])
    finalLabel = max(labelMap.values()) + 1
    newLabels[newLabels == -1] = finalLabel

    print(f'newLabels = {newLabels}')
    splitClusterSet.labels_ = newLabels

    indices999 = np.where(splitClusterSet.labels_ == finalLabel)[0]
    print(f'Indices 999: {indices999}')
    newClusterAB = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA[indices999])
    newClusterSet = np.concatenate((splitClusterSet.cluster_centers_, newClusterAB.cluster_centers_), axis=0)
    mergedClusterSet = KMeans(n_clusters=k-1, n_init=10, random_state=0)
    mergedClusterSet.cluster_centers_ = splitClusterSet
    mergedClusterSet.labels_ = newLabels

    print(f'labels_ : {mergedClusterSet.labels_}')
    print(f'cluster centers: {mergedClusterSet.cluster_centers_}')
    print(f'Clusters {clusterA} and {clusterB} merged.')

    
    
    """
also in bisecting k means:
 Measure the distance for each intra cluster (SSE).
5. Select the cluster that have the largest SSE and
split it to 2 clusters using k-means.
"""

    #newClusterSet = KAlgorithm(newClusterSet, k)
    #print("K algorithm successfully run for this repeat")

    #calculate weights between data points in each cluster
    Enew = calcE(mergedClusterSet, k)
    print(f'E after splitting: {Enew}')
    SSEnew = calcSSE(mergedClusterSet)
    print(f'SSE after splitting: {SSEnew}')

    if (Enew > begE):
        print(f'New value of E ({Eew}) is greater than current value ({begE}). Update solution')
        E = Enew
        clusterSet = mergedClusterSet
    elif (SSEnew < begSSE):
        print(f'Beginning SSE ({begSSE}) is smaller than current value ({SSEnew}). Update solution')
        E = Enew
        clusterSet = mergedClusterSet
    else:
        print("New value of E is less than current value. SSE is greater now. Discard solution.")

    
R = 10
print(f'R set to {R}')
clusterSet = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA)
E = 0.0
print("Cluster set and E variables initialised")

silhouetteScores5 = []
print("silhouette score list initialised")
chScores5 = []
print("calinski harabasz index list initialised")
dbiScores5 = []
print("davies-bouldin index list initialised")
sseScores5 = []
print("sse score list initialised")

for k in K:
    clusterSet = KMeans(n_clusters=k, n_init=10, random_state=0).fit(XPCA)
    print(f'Cluster set for k={k} created')
    for i in range(1, R):
        print(f'R = {i}')
        #perform MAlgorithm passing cluster set to it
        MAlgorithm(clusterSet, k)
        print("M Algorithm successfully performed")

    #silhouette = silhouette_score(XPCA, clusterSet.labels_)
    #print(f'Silhouette Average: {silhouette}')
    #silhouetteScores3.append(sihouette)
    chIndex = calinski_harabasz_score(XPCA, clusterSet.labels_)
    print(f'Calinski Harabasz Score: {chIndex}')
    chScores5.append(chIndex)
    dbiIndex = davies_bouldin_score(XPCA, clusterSet.labels_)
    dbiScores5.append(dbiIndex)
    print('Davies-Bouldin Index:', dbiIndex)
    sse = clusterSet.inertia_
    print(f'SSE Score: {sse}')
    sseScores5.append(sse)
    
    labels0 = clusterSet.labels_[mask0]
    labels1 = clusterSet.labels_[mask1]

    plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
    plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

    plt.xlabel('Primary diagnosis')
    plt.ylabel('Secondary diagnosis')
    plt.title(str(k) + ' Clusters (M-Algorithm)')

    filename = './malg/m_algorithm_' + str(k) + '_clusters.png'
    plt.savefig(filename)
    plt.show()
    
