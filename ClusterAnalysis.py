import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import random
import re
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from math import sqrt

#---------------------------------------------------------------------------------------------
#Data cleaning

#import the patients CSV file into a DataFrame
patients = pd.read_csv('./mimic-iii/PATIENTS.csv')

#import the diagnoses CSV file into a DataFrame
diagnoses = pd.read_csv('./mimic-iii/DIAGNOSES_ICD.csv')

#remove all ICD9 codes that aren't exclusively numeric
diagnoses['ICD9_CODE'] = diagnoses['ICD9_CODE'].astype(str)
diagnoses = diagnoses[diagnoses['ICD9_CODE'].apply(lambda x: re.match(r'^\d+$', x) is not None)]

#removing ICD9 codes 290-319, 630-679, 740-100000
diagnoses['ICD9_CODE'] = diagnoses['ICD9_CODE'].astype(int)
diagnoses = diagnoses[~diagnoses['ICD9_CODE'].between(29000, 31999)]    #mental disorders
diagnoses = diagnoses[~diagnoses['ICD9_CODE'].between(63000, 67900)]    #complications in pregnancy
#congenital abnormalities, perinatal conditions, symptoms, signs, ill-defined conditions, injury and poisoning
diagnoses = diagnoses[~diagnoses['ICD9_CODE'].between(74000, 100000)]
diagnoses['ICD9_CODE'] = diagnoses['ICD9_CODE'].astype(str)

#remove admissions without a primary diagnosis
diagnoses = diagnoses.groupby('HADM_ID').filter(lambda x: (x['SEQ_NUM'] == 1.0).any())

#split into primary/secondary diagnoses DataFrames
diagnosesPrimary = diagnoses[diagnoses['SEQ_NUM'] == 1.0].copy()
diagnosesSec = diagnoses[diagnoses['SEQ_NUM'] != 1.0].copy()

#group icd9 codes by admission for both primary and secondary diagnoses
diagnosesPrimary.drop(columns=['ROW_ID', 'SEQ_NUM'], inplace=True)
diagnosesPrimary.drop_duplicates(inplace=True)
diagnosesPrimary['ICD9_CODE'] = diagnosesPrimary.groupby(['HADM_ID'])['ICD9_CODE'].transform(lambda x: ','.join(x))
diagnosesPrimary.drop_duplicates(inplace=True)

diagnosesSec.drop(columns=['ROW_ID', 'SEQ_NUM'], inplace=True)
diagnosesSec.drop_duplicates(inplace=True)
diagnosesSec['ICD9_CODE'] = diagnosesSec.groupby(['HADM_ID'])['ICD9_CODE'].transform(lambda x: ','.join(x))
diagnosesSec.drop_duplicates(inplace=True)

#keep only SUBJECT_ID and EXPIRE_FLAG
#Note: EXPIRE_FLAG is an indicator for discharged(0)/deceased(1) patient
patients.drop(columns=['ROW_ID', 'GENDER', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN'], inplace=True)

#append the EXPIRE_FLAG column to the primary diagnoses DataFrame
outputData = pd.merge(patients, diagnosesPrimary, on='SUBJECT_ID')
outputData.drop_duplicates(inplace=True)

#add the secondary diagnoses strings to the DataFrame
outputData = pd.merge(outputData, diagnosesSec, on='HADM_ID')
outputData.drop_duplicates(inplace=True)
outputData.rename(columns={'ICD9_CODE_x':'PRIMARY_DIAGNOSIS', 'ICD9_CODE_y':'SECONDARY_DIAGNOSES', 'SUBJECT_ID_x':'SUBJECT_ID'}, inplace=True)
outputData.drop(columns=['SUBJECT_ID_y'], inplace=True)
outputData.drop_duplicates(inplace=True)
outputData['PRIMARY_DIAGNOSIS'] = outputData['PRIMARY_DIAGNOSIS'].astype(int)

#move the EXPIRE_FLAG to the end of the DataFrame, for ease of access later on
expireFlag = outputData.pop('EXPIRE_FLAG')
outputData.insert(len(outputData.columns), 'EXPIRE_FLAG', expireFlag)

outputData.to_csv('./data/ADMISSIONS_PRIMARY_SECONDARY.csv')

#split the secondary diagnoses string, and duplicate the row n times for n diagnoses
#resulting in all primary-secondary diagnosis pairs in the data
diag = outputData.set_index(['SUBJECT_ID', 'HADM_ID', 'PRIMARY_DIAGNOSIS', 'EXPIRE_FLAG'])['SECONDARY_DIAGNOSES']\
            .str.split(',', expand=True)\
            .stack()\
            .reset_index(name='SECONDARY_DIAGNOSIS')\
            .drop('level_4', axis=1)

# merge the original DataFrame with the resulting DataFrame to get the final output
diagData = outputData.merge(diag, on=['SUBJECT_ID', 'HADM_ID', 'PRIMARY_DIAGNOSIS', 'EXPIRE_FLAG'])
diagData = diagData.drop('SECONDARY_DIAGNOSES', axis=1)
diagData['SECONDARY_DIAGNOSIS'] = diagData['SECONDARY_DIAGNOSIS'].astype(int)
diagData.drop_duplicates(inplace=True)

diagData.to_csv('./data/DATA.csv')

#---------------------------------------------------------------------------------------------
#Cluster analysis preparation

#create a feature matrix with primary and secondary diagnoses as features
X = diagData[['PRIMARY_DIAGNOSIS', 'SECONDARY_DIAGNOSIS']].to_numpy()
print(X.shape)

#count how many instances of each ICD9 code in the primary diagnosis column
primCount = outputData['PRIMARY_DIAGNOSIS'].value_counts(normalize=False)

#count how many instances of each ICD9 code in the secondary diagnosis column
secCount = diagData['SECONDARY_DIAGNOSIS'].value_counts(normalize=False)

#perform principal component analysis
pca = PCA(n_components=2)
XPCA = pca.fit_transform(X)
#print(XPCA.shape)

#add the EXPIRE_FLAG column to the array for relative risk calculations later
X = diagData[['PRIMARY_DIAGNOSIS', 'SECONDARY_DIAGNOSIS', 'EXPIRE_FLAG']].to_numpy()

K = [10, 40, 80, 100, 150, 200, 250, 300, 400, 500]
batchSize = [100, 500, 1000, 5000]

#create a custom colormap for up to 600 unique colours, and another with only two
cmap = ListedColormap(np.random.rand(600,3))
colours = ['green', 'red']
cmap2 = ListedColormap(colours)

#create masks for plotting data according to EXPIRE_FLAG values
mask0 = diagData['EXPIRE_FLAG'] == 0
mask1 = diagData['EXPIRE_FLAG'] == 1

#apply masks to the data to produce two numpy arrays for plotting    
X0 = X[mask0]
X1 = X[mask1]

#create a scatter plot to show the distribution of discharged/diseased patients in the data
legendLabels = ['Discharged', 'Deceased']
plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], s=10, cmap=cmap2)
plt.xlabel('Primary diagnosis')
plt.ylabel('Secondary diagnosis')
plt.title('Discharged vs deceased patients')
legend = plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label) for color, label in zip(colours, legendLabels)], title='Patient Outcome', loc='upper left', bbox_to_anchor=(1, 1))

filename = 'diseased_vs_discharged.png'
plt.savefig(filename, bbox_inches='tight')
plt.show()

#---------------------------------------------------------------------------------------------

#creating metric calculation functions

"""
Function to calculate the co-occurrence correlation for two ICD-9 codes
Adapted to be doubled if the admission in which these diagnoses were made resulted in a death

:param codeA: the primary diagnosis ICD-9 code (type: int)
:param codeB: the secondary diagnosis ICD-9 code (type: int)
:param eventFlag: the EXPIRE_FLAG value, 0 or 1 (type: int)
:return relativeRisk: relative risk calculation result (type: float)
"""
def RelativeRisk(codeA, codeB, eventFlag):
    countA = primCount.loc[codeA] if codeA in primCount.index else 0
    countB = secCount.loc[codeB] if codeB in secCount.index else 0
    countAB = countA + countB
    relativeRisk = (countAB * sqrt(2)) / sqrt(countA**2 + countB**2)
    if eventFlag == 1:
        relativeRisk *= 2
    return relativeRisk

"""
Function to calculate the relative risk for every data point within a cluster, and return their sum

:param pClusterSet: the cluster set that the cluster in question belongs to (type: object)
:param pClusterIndex: the index of the cluster that in question (type: int)
:return clusterTotal: the sum of all relative risks for points in the cluster (type: float)
"""
def ClusterRRTotal(pClusterSet, pClusterIndex):
    # Get the indices of the data points in the current cluster
    indices = np.where(pClusterSet.labels_ == pClusterIndex)[0]
    clusterTotal = 0.0
    # Loop through all possible pairs of data points in the cluster
    for i in range(len(indices)):
        # Calculate the relative risk for the pair of data points
        primDiag = X[indices[i], 0]
        secDiag = X[indices[i], 1]
        eventFlag = X[indices[i], 2]
        relativeRisk = RelativeRisk(primDiag, secDiag, eventFlag)
        clusterTotal += relativeRisk
    return clusterTotal


#---------------------------------------------------------------------------------------------

#K-Means

chScores1 = []
dbiScores1 = []
sseScores1 = []
rrScores1 = []

for k in K:
    # perform K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(XPCA)

    chIndex = calinski_harabasz_score(XPCA, kmeans.labels_)
    chScores1.append(chIndex)
    print('Calinski-Harabasz Index:', chIndex)

    dbiIndex = davies_bouldin_score(XPCA, kmeans.labels_)
    dbiScores1.append(dbiIndex)
    print('Davies-Bouldin Index:', dbiIndex)

    sse = kmeans.inertia_
    sseScores1.append(sse)
    print('SSE:', sse)

    rr = 0
    for cluster in range(k):
        rr += ClusterRRTotal(kmeans, cluster)
    rrScores1.append(rr)
    print('Relative risk:', rr)

    #apply the masks to the model's labels
    labels0 = kmeans.labels_[mask0]
    labels1 = kmeans.labels_[mask1]

    #plot the discharged(X0), then diseased(X1) patients
    plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
    plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

    plt.xlabel('Primary diagnosis')
    plt.ylabel('Secondary diagnosis')
    plt.title(str(k) + ' Clusters (k-Means)')

    filename = './kmeans/kmeans_' + str(k) + '_clusters.png'
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()

#plotting the four metrics as line plots
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
plt.show()

#---------------------------------------------------------------------------------------------

#mini batch kmeans

for b in batchSize:
    
    chScores2 = []
    dbiScores2 = []
    sseScores2 = []
    cciScores2 = []
    rrScores2 = []
    
    for k in K:
        # perform clustering
        mbk = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=b, random_state=0)
        mbk.fit(XPCA)

        chIndex = calinski_harabasz_score(XPCA, mbk.labels_)
        chScores2.append(chIndex)
        print('Calinski-Harabasz Index:', chIndex)

        dbiIndex = davies_bouldin_score(XPCA, mbk.labels_)
        dbiScores2.append(dbiIndex)
        print('Davies-Bouldin Index:', dbiIndex)

        sse = mbk.inertia_
        sseScores2.append(sse)
        print('SSE:', sse)

        rr = 0
        for cluster in range(k):
            rr += ClusterRRTotal(mbk, cluster)
        rrScores2.append(rr)
        print('Relative risk:', rr)

        #apply the masks to the model's labels
        labels0 = mbk.labels_[mask0]
        labels1 = mbk.labels_[mask1]

        #plot the discharged(X0), then diseased(X1) patients
        plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
        plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

        plt.xlabel('Primary diagnosis')
        plt.ylabel('Secondary diagnosis')
        plt.title(str(k) + ' Clusters (Mini Batch k-Means, batch size ' + str(b) + ')')

        filename = './mbk/mbk_' + str(k) + '_clusters_' + str(b) + 'batchsize.png'
        plt.savefig(filename, bbox_inches='tight')
        #plt.show()

    #plotting the four metrics as line plots 
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

    ax[0,0].plot(K, chScores2)
    ax[0,1].plot(K, dbiScores2)
    ax[1,0].plot(K, sseScores2)
    ax[1,1].plot(K, rrScores2)

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


#---------------------------------------------------------------------------------------------

#M-Algorithm

"""
Function to calculate the total relative risk for every cluster in the set

:param pClusterSet: the cluster set being calculated (type: object)
:param k: the number of clusters (type: int)
:return tempE: the sum of all clusters' total relative risks (type: float)
"""
def calcE(pClusterSet, k):
    tempE = 0.0
    for i in range(k):
        clusterTotal = ClusterRRTotal(pClusterSet, i)
        tempE += clusterTotal
    return tempE

"""
Function to manually calculate the SSE for a cluster set

:param pClusterSet: the cluster set being calculated (type: object)
:return sse: the sum of SSE values for each cluster in the set (type: float)
"""
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

"""
Function to split one cluster into two. The selected cluster is the one with the greatest SSE value

:param pClusterSet: the cluster set being optimised (type: object)
:return pClusterSet: the cluster set in which one cluster has been split into two (type: object)
"""
def GrowCluster(pClusterSet):
    labels = pClusterSet.labels_.copy()
    centers = pClusterSet.cluster_centers_

    #calculate the SSE for each cluster in the set
    ssePerCluster = np.zeros(pClusterSet.n_clusters)
    for i in range(len(XPCA)):
        center = centers[labels[i]]
        dist = euclidean(XPCA[i], center)
        squaredDist = dist**2
        ssePerCluster[labels[i]] += squaredDist

    #pick the cluster with the largest SSE for splitting
    clusterC = np.argmax(ssePerCluster)
    indicesC = np.where(labels == clusterC)[0]

    #create two clusters from the data points in clusterC   
    newClusterC = KMeans(n_clusters=2, n_init=10, random_state=0).fit(XPCA[indicesC])

    #combine the cluster centers
    pClusterSetCenters = np.concatenate((pClusterSet.cluster_centers_, newClusterC.cluster_centers_), axis=0)
    tempKMeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    tempKMeans.cluster_centers_ = pClusterSetCenters

    #list the data points in the two split clusters
    indicesC = np.where(newClusterC == 0)[0]
    indicesD = np.where(newClusterC == 1)[0]

    #assign new cluster labels to the two split clusters
    labels[indicesC] = k-1
    labels[indicesD] = k
    pClusterSet.labels_ = labels
    
    return pClusterSet

"""
Function for the M-Algorithm
First, the SSE and relative risk (E) are calculated
Then, select the cluster with the largest SSE and split it into two
Then, select two random clusters and merge them
Keep the new cluster set if the SSE is smaller, else if E is larger
Otherwise, discard it

:param pClusterSet: the cluster set being optimised (type: object)
:param k: the number of clusters (type: int)
:return clusterSet: the final cluster set (optimised or original) (type: object)
"""
def MAlgorithm(pClusterSet, k):
    newClusterSet = pClusterSet
    begE = calcE(newClusterSet, k)
    begSSE = calcSSE(newClusterSet)

    splitClusterSet = GrowCluster(newClusterSet)

    #pick two random clusters that have a probability greater than the threshold
    clusterA = 0
    clusterB = 0
    threshold = 2 * 1/k 
    thresholdMet = False
    while thresholdMet == False:
        clusterA = random.randint(0, splitClusterSet.n_clusters)
        clusterB = random.randint(0, splitClusterSet.n_clusters)
        if clusterA == clusterB:
            clusterB += 1

        clusterARR = ClusterRRTotal(splitClusterSet, clusterA)
        clusterBRR = ClusterRRTotal(splitClusterSet, clusterB)
        probability = (clusterARR + clusterBRR) / begE
        
        if probability > threshold:
            thresholdMet = True

    #merge the clusters' labels under a new label, 999
    newLabels = splitClusterSet.labels_.copy()
    newLabels = np.where(newLabels == clusterA, 999, newLabels)
    newLabels = np.where(newLabels == clusterB, 999, newLabels)

    # delete the original clusters from the labels 
    newLabels = np.delete(newLabels, np.where(newLabels == clusterA))
    newLabels = np.delete(newLabels, np.where(newLabels == clusterB))

    #update the cluster labels so there are no gaps, and replace 999 with the final label value
    uniqueLabels = np.unique(newLabels[newLabels != 999])
    labelMap = {label: i for i, label in enumerate(uniqueLabels)}
    newLabels = np.array([labelMap[label] if label != 999 else -1 for label in newLabels])
    finalLabel = max(labelMap.values()) + 1
    newLabels[newLabels == -1] = finalLabel

    splitClusterSet.labels_ = newLabels

    #update the cluster centers for the merged cluster
    indices999 = np.where(splitClusterSet.labels_ == finalLabel)[0]
    newClusterAB = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA[indices999])
    newClusterSet = np.concatenate((splitClusterSet.cluster_centers_, newClusterAB.cluster_centers_), axis=0)
    mergedClusterSet = KMeans(n_clusters=k, n_init=10, random_state=0)
    mergedClusterSet.cluster_centers_ = newClusterSet
    mergedClusterSet.labels_ = newLabels

    Enew = calcE(mergedClusterSet, k)
    SSEnew = calcSSE(mergedClusterSet)
    
    if (SSEnew < begSSE):
        clusterSet = mergedClusterSet
    elif (Enew > begE):
        clusterSet = mergedClusterSet
    else:
        clusterSet = pClusterSet
    
R = 11 #(run 10 times)
clusterSet = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA)

chScores3 = []
dbiScores3 = []
sseScores3 = []
cciScores3 = []
rrScores3 = []

for k in K:
    clusterSet = KMeans(n_clusters=k, n_init=10, random_state=0).fit(XPCA)
    for i in range(1, R):
        #perform MAlgorithm
        MAlgorithm(clusterSet, k)

    chIndex = calinski_harabasz_score(XPCA, clusterSet.labels_)
    chScores3.append(chIndex)
    print(f'Calinski Harabasz Score: {chIndex}')
    
    dbiIndex = davies_bouldin_score(XPCA, clusterSet.labels_)
    print('Davies-Bouldin Index:', dbiIndex)
    dbiScores3.append(dbiIndex)
    
    sse = clusterSet.inertia_
    sseScores3.append(sse)
    print(f'SSE Score: {sse}')

    rr = 0
    for cluster in range(k):
        rr += ClusterRRTotal(clusterSet, cluster)
    rrScores3.append(rr)
    print('Relative risk:', rr)

    #apply the masks to the model's labels
    labels0 = clusterSet.labels_[mask0]
    labels1 = clusterSet.labels_[mask1]

    #plot the discharged(X0), then diseased(X1) patients
    plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
    plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

    plt.xlabel('Primary diagnosis')
    plt.ylabel('Secondary diagnosis')
    plt.title(str(k) + ' Clusters (M Algorithm)')

    filename = './malg/malg_' + str(k) + '_clusters.png'
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()

#plotting the four metrics as line plots 
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
    
plt.show()

#---------------------------------------------------------------------------------------------

#saving the clusters

#optimal value of k selected is 80
finalKMeans = KMeans(n_clusters=80, n_init=10, random_state=0).fit(XPCA)
clusteredLabels = finalKMeans.labels_

clusteredData = diagData.copy()

#assign cluster label to each diagnosis pair
clusteredData['CLUSTER'] = clusteredLabels
print(clusteredData.head())

#create a DataFrame for the codes in each cluster
clusteredCodeData = clusteredData.drop(columns={'SUBJECT_ID', 'HADM_ID', 'EXPIRE_FLAG'})
clusteredCodeData = clusteredData.groupby('CLUSTER').agg({
    'PRIMARY_DIAGNOSIS': lambda x: ','.join(x.astype(str)),
    'SECONDARY_DIAGNOSIS': lambda x: ','.join(x.astype(str))
})
#remove duplicate codes from each string
clusteredCodeData['PRIMARY_DIAGNOSIS'] = clusteredCodeData['PRIMARY_DIAGNOSIS'].apply(lambda x: ','.join(sorted(list(set([s.strip() for s in x.split(',')])), key=str)))
clusteredCodeData['SECONDARY_DIAGNOSIS'] = clusteredCodeData['SECONDARY_DIAGNOSIS'].apply(lambda x: ','.join(sorted(list(set([s.strip() for s in x.split(',')])), key=str)))

clusteredCodeData.to_csv('./data/CLUSTERED_CODES_DATA.csv')

#create a DataFrame showing which clusters each admission belongs to
clusteredAdmData = clusteredData.merge(outputData, on=['SUBJECT_ID', 'HADM_ID', 'PRIMARY_DIAGNOSIS', 'EXPIRE_FLAG'])
clusteredAdmData.drop(columns={'SECONDARY_DIAGNOSIS'}, inplace=True)
clusteredAdmData = clusteredAdmData.groupby(['SUBJECT_ID', 'HADM_ID', 'PRIMARY_DIAGNOSIS', 'EXPIRE_FLAG', 'SECONDARY_DIAGNOSES'])['CLUSTER'].apply(lambda x: ','.join(x.astype(str))).reset_index()
clusteredAdmData.drop_duplicates(inplace=True)

clusteredAdmData.to_csv('./data/CLUSTERED_ADMISSION_DATA.csv')




