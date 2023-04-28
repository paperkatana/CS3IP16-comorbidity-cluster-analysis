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
from comorbidipy import comorbidity
from math import sqrt

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

#removing ICD9 codes 290-319, 630-679, 740-799
diagnoses['ICD9_CODE'] = diagnoses['ICD9_CODE'].astype(int)
diagnoses = diagnoses[~diagnoses['ICD9_CODE'].between(290, 319)]    #mental disorders
diagnoses = diagnoses[~diagnoses['ICD9_CODE'].between(630, 679)]    #complications in pregnancy
diagnoses = diagnoses[~diagnoses['ICD9_CODE'].between(740, 999)]    #congenital abnormalities, perinatal conditions, symptoms, signs, ill-defined conditions, injury and poisoning
diagnoses['ICD9_CODE'] = diagnoses['ICD9_CODE'].astype(str)

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
print(diagData.shape)
diagData.to_csv('./data/DATA.csv')

def appendPrimaryDiagnosis(row):
    primDiag = str(row['PRIMARY_DIAGNOSIS'])
    secDiag = row['SECONDARY_DIAGNOSES']
    return secDiag + ',' + primDiag

cciData = outputData.copy()

cciData['DIAGNOSES'] = cciData.apply(appendPrimaryDiagnosis, axis=1)
print(cciData.head())
cciData.to_csv('./data/CCIDATA.csv')

#---------------------------------------------------------------------------------------------
#Cluster analysis preparation

#cci = comorbidity(cciData, id='SUBJECT_ID', code='DIAGNOSES', age=None, score='charlson', icd='icd9', variant='quan', weighting='quan', assign0=True)
#print(cci.head())
#cci.to_csv('./data/CCI.csv')

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
batchSize = [100, 500, 1000, 5000]
print("values of k set")

#create a custom colormap for up to 600 unique colours
cmap = ListedColormap(np.random.rand(600,3))
colours = ['green', 'red']
cmap2 = ListedColormap(colours)

mask0 = diagData['EXPIRE_FLAG'] == 0
mask1 = diagData['EXPIRE_FLAG'] == 1
    
X0 = X[mask0]
X1 = X[mask1]

"""
labels = ['Discharged', 'Deceased']
plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], s=10, cmap=cmap2)
plt.xlabel('Primary diagnosis')
plt.ylabel('Secondary diagnosis')
plt.title('Discharged vs deceased patients')
legend = plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label) for color, label in zip(colours, labels)], title='Patient Outcome', loc='upper left', bbox_to_anchor=(1, 1))

filename = 'diseased_vs_discharged.png'
plt.savefig(filename, bbox_inches='tight')
#plt.show()
"""
#---------------------------------------------------------------------------------------------

#creating metric calculation functions

#use this as a metric on whether the clusters demonstrate good comorbidity in the report
def calcCCI(pLabels):
    if 'CLUSTER_ID' in cciData.columns:
        cciData.drop('CLUSTER_ID', axis=1, inplace=True)

    # Create a dictionary to map patient IDs to cluster assignments
    idToCluster = dict(zip(cciData['SUBJECT_ID'], pLabels))
        
    # Create a new column in diagData to represent the cluster assignment for each patient
    cciData['CLUSTER_ID'] = cciData['SUBJECT_ID'].map(idToCluster)

    # Calculate the CCI for each cluster
    clusterCCI = cciData.groupby('CLUSTER_ID').apply(lambda x: comorbidity(x, id='SUBJECT_ID', code='DIAGNOSES', age=None, score='charlson', icd='icd9', variant='quan', weighting='quan', assign0=True)['comorbidity_score'].sum())
    #print(clusterCCI)
    return clusterCCI

def RelativeRisk(codeA, codeB, eventFlag):
    #print("Beginning RelativeRisk(codeA, codeB)")
    #print(f'Primary diagnosis: {codeA}. Secondary diagnosis: {codeB}.')
    #countTotal = primCount.sum() + secCount.sum()
    #print(f'Count total: {countTotal}. primCount={primCount.sum()} and secCount={secCount.sum()}')
    countA = primCount.loc[codeA] if codeA in primCount.index else 0
    #print(f'Count of primary diagnosis {codeA}: {countA}')
    countB = secCount.loc[codeB] if codeB in secCount.index else 0
    #print(f'Count of secondary diagnosis {codeB}: {countB}')
    countAB = countA + countB
    #print(f'count A + count B = {countAB}')
    #relativeRisk = (countAB / countTotal) / ((countA / countTotal)*(countB / countTotal))
    relativeRisk = (countAB * sqrt(2)) / sqrt(countA**2 + countB**2)
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

    #clusterCCI = calcCCI(pClusterSet.labels_)
    #clusterIndexCCI = clusterCCI[pClusterIndex] #this is the cluster's CCI 
    
    #print("Looped through all possible pairs of data points in the cluster and calculated relative risk")
    #print(f'Final cluster total for cluster {pClusterIndex}: {clusterTotal}')
    #print("Ending ClusterRRTotal(clusterIndex)")
    return clusterTotal

#---------------------------------------------------------------------------------------------
"""
#K-Means
   
    #totalCCI = cluster_cci.sum()
    #print(f'Total CCI for this cluster set: {totalCCI}')
    #return totalCCI

chScores1 = []
dbiScores1 = []
sseScores1 = []
cciScores1 = []
rrScores1 = []

for k in K:
    print(f'k = {k}')

    # perform K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(XPCA)
    print("kmeans fit to k")

    #calinski-harabasz index (also known as the Variance Ratio Criterion)
    #aiming for a higher value
    chIndex = calinski_harabasz_score(XPCA, kmeans.labels_)
    chScores1.append(chIndex)
    print('Calinski-Harabasz Index:', chIndex)

    #davies-bouldin index 
    #aiming for a lower value
    dbiIndex = davies_bouldin_score(XPCA, kmeans.labels_)
    dbiScores1.append(dbiIndex)
    print('Davies-Bouldin Index:', dbiIndex)

    #sse
    #aiming for a lower value
    #The inertia_ attribute returns the sum of squared distances of all data points to their closest centroid
    sse = kmeans.inertia_
    sseScores1.append(sse)
    print('SSE:', sse)

    cci = calcCCI(kmeans.labels_)
    cciScores1.append(cci)
    #print(cci)

    rr = 0
    for cluster in range(k):
        rr += ClusterRRTotal(kmeans, cluster)
    rrScores1.append(rr)
    print(rr)
    
    labels0 = kmeans.labels_[mask0]
    labels1 = kmeans.labels_[mask1]

    plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
    plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

    plt.xlabel('Primary diagnosis')
    plt.ylabel('Secondary diagnosis')
    plt.title(str(k) + ' Clusters (k-Means)')

    filename = './kmeans/kmeans_' + str(k) + '_clusters.png'
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

ax[0,0].plot(K, chScores1)
ax[0,1].plot(K, dbiScores1)
ax[1,0].plot(K, sseScores1)
ax[1,1].plot(K, rrScores1)

ax[0,0].set_xlabel('Number of clusters')
ax[0,0].set_ylabel('Calinki-Harabasz Index')
ax[0,0].set_title('Calinski-Harabasz Scores')
ax[0,1].set_xlabel('Number of clusters')
ax[0,1].set_ylabel('Davies-Bouldin Index')
ax[0,1].set_title('Davies-Bouldin Scores')
ax[1,0].set_xlabel('Number of clusters')
ax[1,0].set_ylabel('Sum of Squared Error Score')
ax[1,0].set_title('SSE Scores')
ax[1,1].set_xlabel('Number of clusters')
ax[1,1].set_ylabel('Relative Risk Score')
ax[1,1].set_title('Relative Risk')

plt.tight_layout()

filename = './kmeans/kmeans_clusters_metrics.png'
plt.savefig(filename, bbox_inches='tight')
#plt.show()

#---------------------------------------------------------------------------------------------

#mini batch kmeans

chScores2 = []
print("calinski harabasz index list initialised")
dbiScores2 = []
print("davies-bouldin index list initialised")
sseScores2 = []
print("sse score list initialised")
cciScores2 = []
rrScores2 = []

for b in batchSize:
    chScoresTemp = []
    dbiScoresTemp = []
    sseScoresTemp = []
    cciScoresTemp = []
    rrScoresTemp = []
    for k in K:
        print(f'k = {k}. batch size = {b}')

        # perform clustering
        mbk = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=b, random_state=0)
        mbk.fit(XPCA)

        #calinski-harabasz index (also known as the Variance Ratio Criterion)
        #aiming for a higher value
        chIndex = calinski_harabasz_score(XPCA, mbk.labels_)
        chScoresTemp.append(chIndex)
        chScores2.append(chIndex)
        print('Calinski-Harabasz Index:', chIndex)

        #davies-bouldin index 
        #aiming for a lower value
        dbiIndex = davies_bouldin_score(XPCA, mbk.labels_)
        dbiScoresTemp.append(dbiIndex)
        dbiScores2.append(dbiIndex)
        print('Davies-Bouldin Index:', dbiIndex)

        #sse
        #aiming for a lower value
        #The inertia_ attribute returns the sum of squared distances of all data points to their closest centroid
        sse = mbk.inertia_
        sseScoresTemp.append(sse)
        sseScores2.append(sse)
        print('SSE:', sse)

        #cci = calcCCI(mbk.labels_)
        #cciScores2.append(cci)
        #print(cci)

        rr = 0
        for cluster in range(k):
            rr += ClusterRRTotal(mbk, cluster)
        rrScoresTemp.append(rr)
        print(rr)

        labels0 = mbk.labels_[mask0]
        labels1 = mbk.labels_[mask1]

        plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
        plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

        plt.xlabel('Primary diagnosis')
        plt.ylabel('Secondary diagnosis')
        plt.title(str(k) + ' Clusters (Mini Batch k-Means, batch size ' + str(b) + ')')

        filename = './mbk/mbk_' + str(k) + '_clusters_' + str(b) + 'batchsize.png'
        plt.savefig(filename, bbox_inches='tight')
        #plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

    ax[0,0].plot(K, chScoresTemp)
    ax[0,1].plot(K, dbiScoresTemp)
    ax[1,0].plot(K, sseScoresTemp)
    ax[1,1].plot(K, rrScoresTemp)

    ax[0,0].set_xlabel('Number of clusters')
    ax[0,0].set_ylabel('Calinki-Harabasz Index')
    ax[0,0].set_title('Calinski-Harabasz Scores')
    ax[0,1].set_xlabel('Number of clusters')
    ax[0,1].set_ylabel('Davies-Bouldin Index')
    ax[0,1].set_title('Davies-Bouldin Scores')
    ax[1,0].set_xlabel('Number of clusters')
    ax[1,0].set_ylabel('Sum of Squared Error Score')
    ax[1,0].set_title('SSE Scores')
    ax[1,1].set_xlabel('Number of clusters')
    ax[1,1].set_ylabel('Relative Risk Score')
    ax[1,1].set_title('Relative Risk')

    plt.tight_layout()

    filename = './mbk/mbk_' + str(b) + '_batch_size_metrics.png'
    plt.savefig(filename, bbox_inches='tight')
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
    labels = pClusterSet.labels_.copy()
    centers = pClusterSet.cluster_centers_

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
    #print("created list of data points for cluster C")

    """
    #find out how many nodes should be in the new cluster (% of original cluster)
    unbalanceFactor = random.uniform(0.05, 0.95)
    print(f'Unbalance Factor {unbalanceFactor} selected')
    targetSize = int(len(indicesC) * unbalanceFactor)
    print(f'Calculated target size: {targetSize}')
    
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
    
    return pClusterSet


#def MergeClusters(pClusterSet, pE, k):

def MAlgorithm(pClusterSet, k):
    print(f'Beginning MAlgorithm(k). k = {k}')
    #create copy of prevaling cluster set as newClusterSet
    newClusterSet = pClusterSet
    begE = calcE(newClusterSet, k)
    begSSE = calcSSE(newClusterSet)

    splitClusterSet = GrowCluster(newClusterSet)
    print("New cluster grown")

    clusterA = 0
    clusterB = 0
    threshold = 2 * 1/k 
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
        clusterBRR = ClusterRRTotal(splitClusterSet, clusterB)
        probability = (clusterARR + clusterBRR) / begE
        #print(f'probability = {clusterARR} + {clusterBRR} / {begE} = {probability}')
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

    splitClusterSet.labels_ = newLabels

    indices999 = np.where(splitClusterSet.labels_ == finalLabel)[0]
    newClusterAB = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA[indices999])
    newClusterSet = np.concatenate((splitClusterSet.cluster_centers_, newClusterAB.cluster_centers_), axis=0)
    mergedClusterSet = KMeans(n_clusters=k, n_init=10, random_state=0)
    mergedClusterSet.cluster_centers_ = newClusterSet
    mergedClusterSet.labels_ = newLabels
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
    SSEnew = calcSSE(mergedClusterSet)

    
    if (SSEnew < begSSE):
        print(f'Beginning SSE ({begSSE}) is smaller than current value ({SSEnew}). Update solution')
        E = Enew
        clusterSet = mergedClusterSet
    elif (Enew > begE):
        print(f'New value of E ({Enew}) is greater than current value ({begE}). Update solution')
        E = Enew
        clusterSet = mergedClusterSet
    else:
        print("New value of E is less than current value. SSE is greater now. Discard solution.")
    
R = 11 #(run 10 times)
clusterSet = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA)
E = 0.0

chScores3 = []
dbiScores3 = []
sseScores3 = []
cciScores3 = []
rrScores3 = []

for k in K:
    clusterSet = KMeans(n_clusters=k, n_init=10, random_state=0).fit(XPCA)
    print(f'Cluster set for k={k} created')
    for i in range(1, R):
        print(f'R = {i}')
        #perform MAlgorithm passing cluster set to it
        MAlgorithm(clusterSet, k)
        print("M Algorithm successfully performed")

    chIndex = calinski_harabasz_score(XPCA, clusterSet.labels_)
    print(f'Calinski Harabasz Score: {chIndex}')
    chScores3.append(chIndex)
    dbiIndex = davies_bouldin_score(XPCA, clusterSet.labels_)
    dbiScores3.append(dbiIndex)
    print('Davies-Bouldin Index:', dbiIndex)
    sse = clusterSet.inertia_
    print(f'SSE Score: {sse}')
    sseScores3.append(sse)

    cci = calcCCI(clusterSet.labels_)
    cciScores3.append(cci)
    #print(cci)

    rr = 0
    for cluster in range(k):
        rr += ClusterRRTotal(clusterSet, cluster)
    rrScores3.append(rr)
    print(rr)
    
    labels0 = clusterSet.labels_[mask0]
    labels1 = clusterSet.labels_[mask1]

    plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
    plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

    plt.xlabel('Primary diagnosis')
    plt.ylabel('Secondary diagnosis')
    plt.title(str(k) + ' Clusters (M Algorithm)')

    filename = './malg/malg_' + str(k) + '_clusters.png'
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

ax[0,0].plot(K, chScores3)
ax[0,1].plot(K, dbiScores3)
ax[1,0].plot(K, sseScores3)
ax[1,1].plot(K, rrScores3)

ax[0,0].set_xlabel('Number of clusters')
ax[0,0].set_ylabel('Calinki-Harabasz Index')
ax[0,0].set_title('Calinski-Harabasz Scores')
ax[0,1].set_xlabel('Number of clusters')
ax[0,1].set_ylabel('Davies-Bouldin Index')
ax[0,1].set_title('Davies-Bouldin Scores')
ax[1,0].set_xlabel('Number of clusters')
ax[1,0].set_ylabel('Sum of Squared Error Score')
ax[1,0].set_title('SSE Scores')
ax[1,1].set_xlabel('Number of clusters')
ax[1,1].set_ylabel('Relative Risk Score')
ax[1,1].set_title('Relative Risk')

plt.tight_layout()

filename = './malg/malg_clusters_metrics.png'
plt.savefig(filename, bbox_inches='tight')
    
#plt.show()

#---------------------------------------------------------------------------------------------
"""
#CLARA

chScores4 = []
dbiScores4 = []
sseScores4 = []
cciScores4 = []
rrScores4 = []

maxNeighbours = [2,5,8,10,12]
numLocal = [3,4,5]

for n in maxNeighbours:
    chScoresTemp = []
    dbiScoresTemp = []
    sseScoresTemp = []
    cciScoresTemp = []
    rrScoresTemp = []    
    for k in K:
        print(f'k = {k}. maxNeighbours = {n}')

        # perform clustering
        clara = clarans(XPCA, k, n, 3)
        print("CLARA created")
        clara.process()
        print("CLARA processed")
        clusters = clara.get_clusters()
        medoids = clara.get_medoids()

        #calinski-harabasz index (also known as the Variance Ratio Criterion)
        #aiming for a higher value
        chIndex = calinski_harabasz_score(XPCA, clusters)
        chScoresTemp.append(chIndex)
        chScores4.append(chIndex)
        print('Calinski-Harabasz Index:', chIndex)
        print(chScoresTemp)

        #davies-bouldin index 
        #aiming for a lower value
        dbiIndex = davies_bouldin_score(XPCA, clusters)
        dbiScoresTemp.append(dbiIndex)
        dbiScores4.append(dbiIndex)
        print('Davies-Bouldin Index:', dbiIndex)

        #sse
        #aiming for a lower value
        #The inertia_ attribute returns the sum of squared distances of all data points to their closest centroid
        distances = cdist(XPCA, medoids, 'euclidean')
        sse = np.min(distances, axis=1).sum()
        sseScoresTemp.append(sse)
        sseScores4.append(sse)
        print('SSE:', sse)

        cci = calcCCI(clusters)
        cciScores4.append(cci)
        #print(cci)

        rr = 0
        for cluster in range(k):
            rr += ClusterRRTotal(clara, cluster)
        rrScores4.append(rr)
        print(rr)

        labels0 = clusters[mask0]
        labels1 = clusters[mask1]

        plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
        plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

        plt.xlabel('Primary diagnosis')
        plt.ylabel('Secondary diagnosis')
        plt.title(str(k) + ' Clusters (CLARA, maxNeighbor=' + str(n) + ')')
        
        filename = './clara/clara_' + str(k) + '_clusters_' + str(n) + 'maxneighbor.png'
        plt.savefig(filename, bbox_inches='tight')
        #plt.show()
        

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

    ax[0,0].plot(K, chScoresTemp)
    ax[0,1].plot(K, dbiScoresTemp)
    ax[1,0].plot(K, sseScoresTemp)
    ax[1,1].plot(K, rrScoresTemp)

    ax[0,0].set_xlabel('Batch Size')
    ax[0,0].set_ylabel('Calinki-Harabasz Index')
    ax[0,0].set_title('Calinski-Harabasz Scores')
    ax[0,1].set_xlabel('Batch Size')
    ax[0,1].set_ylabel('Davies-Bouldin Index')
    ax[0,1].set_title('Davies-Bouldin Scores')
    ax[1,0].set_xlabel('Batch Size')
    ax[1,0].set_ylabel('Sum of Squared Error Score')
    ax[1,0].set_title('SSE Scores')
    ax[1,1].set_xlabel('Batch Size')
    ax[1,1].set_ylabel('Relative Risk Score')
    ax[1,1].set_title('Relative Risk')

    plt.tight_layout()

    filename = './clara/clara_' + str(n) + '_max_neighbours_metrics.png'
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()

"""
