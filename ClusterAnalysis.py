import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import random
import re
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
outputData.drop(columns=['SUBJECT_ID_y'], inplace=True)
outputData.drop_duplicates(inplace=True)
outputData['PRIMARY_DIAGNOSIS'] = outputData['PRIMARY_DIAGNOSIS'].astype(int)
#print(outputData.head())

outputData.to_csv('./data/ADMISSIONS_PRIMARY_SECONDARY.csv')

diag = outputData.set_index(['SUBJECT_ID', 'EXPIRE_FLAG', 'HADM_ID', 'PRIMARY_DIAGNOSIS'])['SECONDARY_DIAGNOSES']\
            .str.split(',', expand=True)\
            .stack()\
            .reset_index(name='SECONDARY_DIAGNOSIS')\
            .drop('level_4', axis=1)

# merge the original dataframe with the resulting dataframe to get the final output
diagData = outputData.merge(diag, on=['SUBJECT_ID', 'EXPIRE_FLAG', 'HADM_ID', 'PRIMARY_DIAGNOSIS'])
diagData = diagData.drop('SECONDARY_DIAGNOSES', axis=1)
diagData['SECONDARY_DIAGNOSIS'] = diagData['SECONDARY_DIAGNOSIS'].astype(int)
diagData.drop_duplicates()
# print the resulting dataframe
print(diagData.head())
diagData.to_csv('./data/DATA.csv')

#---------------------------------------------------------------------------------------------
#Cluster analysis preparation

#colidx = df.columns.get_loc('EXPIRE_FLAG')

# create a feature matrix with primary and secondary diagnoses as features
X = diagData[['PRIMARY_DIAGNOSIS', 'SECONDARY_DIAGNOSIS']].to_numpy()
print(X.shape)

#...
"""
mlb = MultiLabelBinarizer()
print("initialised multi label binarizer")
mlbData = mlb.fit_transform(df['SECONDARY_DIAGNOSES'])
print("fitted multi label binarizer to seconday diagnoses")
print(mlbData.shape)
"""

#count how many instances of each ICD9 code in the primary diagnosis column
primCount = outputData['PRIMARY_DIAGNOSIS'].value_counts(normalize=False)
#print("Primary diagnoses counted")

#count how many instances of each ICD9 code in the secondary diagnosis column
secCount = diagData['SECONDARY_DIAGNOSIS'].value_counts(normalize=False)
#print("Secondary diagnoses counted")


# Print the counts for each code
"""for code, count in primCount.items():
    print(f"{code}: {count}")

# Get the list of codes for each row of mlbData
codeList = mlb.inverse_transform(mlbData)
print(codeList)
# Create a flattened list of all codes
secCodeList = [code for codes in codeList for code in codes]

outputData['SECONDARY_DIAGNOSES'] = outputData['SECONDARY_DIAGNOSES'].str.split(',')
# Count the occurrences of each code
secCount = Counter(secCodeList)
print("Secondary diagnoses counted")
# Print the counts for each code
for code, count in secCount.items():
    print(f"{code}: {count}")
"""
#perform principal component analysis
pca = PCA(n_components=2)
#print("initialised pca")
XPCA = pca.fit_transform(X)
#print("fitted pca to multi label binarized data")
#print(XPCA.shape)

"""
tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000)
print("created tsne object")
xtsne = tsne.fit_transform(X)
print("tsne object fit to xpca data")
"""

K = [4]
#K = [10, 40, 80, 100, 150, 200, 250, 300, 400, 500, 600]
print("values of k set")

#create a custom colormap for up to 600 unique colours
cmap = ListedColormap(np.random.rand(600,3))

mask0 = diagData['EXPIRE_FLAG'] == 0
mask1 = diagData['EXPIRE_FLAG'] == 1
    
X0 = X[mask0]
X1 = X[mask1]
#---------------------------------------------------------------------------------------------

#K-Means

"""
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

    
    #silhouette score
    #aiming for as close to 1.0 as possible
    silhouette = silhouette_score(XPCA, kmeans.labels_)
    silhouetteScores1.append(silhouette)
    print("The average silhouette_score is :", silhouette)
    

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
    
    labels0 = kmeans.labels_[mask0]
    labels1 = kmeans.labels_[mask1]

    plt.scatter(X0[:, 0], X0[:, 1], c=labels0, marker='o', s=10, cmap=cmap)
    plt.scatter(X1[:, 0], X1[:, 1], c=labels1, marker='x', s=10, cmap=cmap)

    plt.xlabel('Primary diagnosis')
    plt.ylabel('Secondary diagnosis')
    plt.title(str(k) + ' Clusters (k-Means)')

    filename = './kmeans/kmeans_' + str(k) + '_clusters.png'
    plt.savefig(filename)
    plt.show()

    #plt.scatter(XPCA[:,0], XPCA[:,1], c=kmeans.labels_, cmap='rainbow')
    #plt.show()
    
    tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000)
    print("created tsne object")
    xtsne = tsne.fit_transform(XPCA)
    print("tsne object fit to xpca data")
    plt.scatter(xtsne[:,0], xtsne[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.show()
    #save model
    
    # visualize results
    # plot the clusters with different markers based on the event column
    
    for i in range(len(XPCA)):
        if df.loc[i, 'EXPIRE_FLAG'] == 0:
            plt.scatter(XPCA[i,0], XPCA[i,1], c=kmeans.labels_[i], marker='o', cmap='rainbow')
        else:
            plt.scatter(XPCA[i,0], XPCA[i,1], c=kmeans.labels_[i], marker='x', cmap='rainbow')

    filename = './kmeans/' + str(k) + '_clusters.png'
    plt.savefig(filename)
    plt.show()
"""    
#---------------------------------------------------------------------------------------------

#K-Algorithm

def RelativeRisk(codeA, codeB):
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
        relativeRisk = RelativeRisk(primDiag, secDiag)
        clusterTotal += relativeRisk
    #print("Looped through all possible pairs of data points in the cluster and calculated relative risk")
    #print(f'Final cluster total: {clusterTotal}')
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



clusterSet = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA)
print("Cluster set initialised with one cluster")

silhouetteScores2 = []
print("silhouette score list initialised")
chScores2 = []
print("calinski harabasz index list initialised")
sseScores2 = []
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
            
def GrowCluster(pClusterSet, pClusterC, pUnbalanceFactor):
    #print("Beginning GrowCluster(pClusterSet, pClusterC, pUnbalanceFactor)")
    print(f'pClusterC: {pClusterC}. pUnbalanceFactor: {pUnbalanceFactor}')
    #get clusterC data
    print(pClusterSet.labels_)
    labels = pClusterSet.labels_.copy()
    indicesC = np.where(labels == pClusterC)[0]
    print(indicesC)
    #print("created list of data points for cluster C")
    
    #find out how many nodes should be in the new cluster (% of original cluster)
    targetSize = int(len(indicesC) * pUnbalanceFactor)
    print(f'Calculated target size: {targetSize}')

    randomPrimIndex = np.random.choice(indicesC)
    indicesD = [randomPrimIndex]
    indicesC = np.delete(indicesC, np.where(indicesC == randomPrimIndex))
    #print("Removed primary diagnosis from cluster C")

    relativeRisks = []
    for index in indicesC:
        relativeRisks.append(RelativeRisk(X[randomPrimIndex, 0], X[index, 1]))
    #print("Created list of relative risks for every data point in cluster C, in comparison to the primary diagnosis in cluster D")
        
    sortedIndicesC = [x for _, x in sorted(zip(relativeRisks, indicesC), reverse=True)]
    indicesD = sortedIndicesC[:targetSize]
    indicesC = sortedIndicesC[targetSize:]
    print(f'IndicesD[0]: {indicesD[0]}. Target Size: {targetSize}. IndicesC[0]: {indicesC[0]}')
        
    newClusterC = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA[indicesC])
    pClusterSet = np.concatenate((pClusterSet.cluster_centers_, newClusterC.cluster_centers_), axis=0)
    tempKMeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    tempKMeans.cluster_centers_ = pClusterSet

    newClusterD = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA[indicesD])
    tempKMeans = np.concatenate((tempKMeans.cluster_centers_, newClusterD.cluster_centers_), axis=0)
    newKMeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    newKMeans.cluster_centers_ = tempKMeans
    
    labels[indicesD] = k
    newKMeans.labels_ = labels
    print(f'Final labels: {newKMeans.labels_}')
    
    return newKMeans


#def MergeClusters(pClusterSet, pE, k):

def MAlgorithm(pClusterSet, k):
    print(f'Beginning MAlgorithm(k). k = {k}')
    #create copy of prevaling cluster set as newClusterSet
    newClusterSet = pClusterSet
    begE = calcE(newClusterSet, k)

    clusterA = 3
    clusterB = 0
    threshold = 1/k
    print(f'Threshold set to {threshold}')
    thresholdMet = False
    while thresholdMet == False:
        clusterA = random.randint(0, k-1)
        print(f'Picked random cluster A: {clusterA}')
        clusterB = random.randint(0, k-1)
        print(f'Picked random cluster B: {clusterB}')
        if clusterA == clusterB:
            clusterB += 1
            print(f'clusters A and B are the same. New cluster B: {clusterB}')
        #calculate sum of their weights
        clusterARR = ClusterRRTotal(newClusterSet, clusterA)
        #print(f'Cluster ARR: {clusterARR}')
        clusterBRR = ClusterRRTotal(newClusterSet, clusterB)
        #print(f'Cluster BRR: {clusterBRR}')
        probability = (clusterARR + clusterBRR) / begE
        print(f'probability = {clusterARR} + {clusterBRR} / {begE} = {probability}')
        if probability > threshold:
            print("Probability is greater than threshold")
            thresholdMet = True

    newLabels = newClusterSet.labels_.copy()
    newLabels = np.where(newClusterSet.labels_ == clusterA, 999, newClusterSet.labels_)
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
    newClusterSet.labels_ = newLabels

    indices999 = np.where(newClusterSet.labels_ == finalLabel)[0]
    print(f'Indices 999: {indices999}')
    newClusterAB = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA[indices999])
    newClusterSet = np.concatenate((pClusterSet.cluster_centers_, newClusterAB.cluster_centers_), axis=0)
    newKMeans = KMeans(n_clusters=k-1, n_init=10, random_state=0)
    newKMeans.cluster_centers_ = newClusterSet
    newKMeans.labels_ = newLabels

    print(f'labels_ : {newKMeans.labels_}')
    print(f'cluster centers: {newKMeans.cluster_centers_}')
    #print(f'Clusters {clusterA} and {clusterB} merged.')
    tempE = calcE(newKMeans, k)
    print(f'E after merging: {tempE}')
    
    #randomly select another cluster for splitting, a cluster seed and unbalanceFactor
    clusterC = random.randint(0, k-2)
    print(f'Cluster {clusterC} selected for splitting')
    #create a boolean array for all values in XPCA where they are in clusterC, then select from these points
    #clusterSeed = random.choice(XPCA[kmeans.labels_ == clusterC])
    unbalanceFactor = random.uniform(0.05, 0.95)
    print(f'Unbalance Factor {unbalanceFactor} selected')

    splitClusterSet = GrowCluster(newKMeans, clusterC, unbalanceFactor)
    print("New cluster grown")
    
    """
also in bisecting k means:
 Measure the distance for each intra cluster (SSE).
5. Select the cluster that have the largest SSE and
split it to 2 clusters using k-means.
"""

    #newClusterSet = KAlgorithm(newClusterSet, k)
    #print("K algorithm successfully run for this repeat")

    #calculate weights between data points in each cluster
    Enew = calcE(splitClusterSet, k)
    print(f'E after splitting: {Enew}')

    if (Enew > begE):
        print(f'New value of E ({Eew}) is greater than current value ({begE}). Update solution')
        E = Enew
        clusterSet = splitClusterSet
    else:
        print("New value of E is less than current value. Discard solution.")

    
R = 10
print(f'R set to {R}')
clusterSet = KMeans(n_clusters=1, n_init=10, random_state=0).fit(XPCA)
E = 0.0
print("Cluster set and E variables initialised")

silhouetteScores3 = []
print("silhouette score list initialised")
chScores3 = []
print("calinski harabasz index list initialised")
sseScores3 = []
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
    chScores3.append(chIndex)
    sse = clusterSet.inertia_
    print(f'SSE Score: {sse}')
    sseScores3.append(sse)
    
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
    
