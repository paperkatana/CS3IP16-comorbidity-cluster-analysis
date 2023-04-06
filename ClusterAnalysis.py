import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.sparse import hstack
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print("all packages imported")
#---------------------------------------------------------------------------------------------

# read in the data
df = pd.read_csv("ADMISSIONS_PRIMARY_SECONDARY.csv")
print("data imported")
print(df.head())

#colidx = df.columns.get_loc('EXPIRE_FLAG')

#count how many instances of each ICD9 code in the primary diagnosis column
icd9CountsPrim = df['PRIMARY_DIAGNOSIS'].value_counts(normalize=False)
print("counted number of primary diagnoses")

#...
mlb = MultiLabelBinarizer()
print("initialised multi label binarizer")
mlbData = mlb.fit_transform(df['SECONDARY_DIAGNOSES'])
print("fitted multi label binarizer to seconday diagnoses")
print(mlbData.shape)

# create a list of all ICD9 codes in the secondary diagnoses column
icd9Codes = list(mlbData.classes_)
print("created a list of all unique secondary diagnoses")

# initialize a dictionary to hold the number of times each secondary diagnosis occurs
icd9CountsSec = {}
print("initialised secondary diagnosis count dictionary")

# loop through each ICD9 code and count how many times it appears in the DataFrame
for code in icd9Codes:
    count = (df['SECONDARY_DIAGNOSES'].apply(lambda x: code in x.split(',')).sum())
    icd9CountsSec[code] = count
print("secondary diagnoses counted")

#perform principal component analysis
pca = PCA(n_components=2)
print("initialised pca")
XPCA = pca.fit_transform(mlbData)
print("fitted pca to multi label binarized data")
print(XPCA.shape)

K = [100]
#K = [4, 6, 8, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500, 600]
print("values of k set")

#---------------------------------------------------------------------------------------------

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
    silhouetteAvg = silhouette_score(XPCA, kmeans.labels_)
    silhouetteScores1.append(silhouetteAvg)
    print("The average silhouette_score is :", silhouetteAvg)

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

    #plt.scatter(XPCA[:,0], XPCA[:,1], c=kmeans.labels_, cmap='rainbow')
    #plt.show()

    tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000)
    print("created tsne object")
    xtsne = tsne.fit_transform(XPCA)
    print("tsne object fit to xpca data")
    plt.scatter(xtsne[:,0], xtsne[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.show()

    # visualize results
    # plot the clusters with different markers based on the event column
    """
    for i in range(len(XPCA)):
        if df.loc[i, 'EXPIRE_FLAG'] == 0:
            plt.scatter(XPCA[i,0], XPCA[i,1], c=kmeans.labels_[i], marker='o', cmap='rainbow')
        else:
            plt.scatter(XPCA[i,0], XPCA[i,1], c=kmeans.labels_[i], marker='x', cmap='rainbow')

    filename = './kmeans/' + str(k) + '_clusters.png'
    plt.savefig(filename)
    plt.show()"""


#---------------------------------------------------------------------------------------------

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

def RelativeRisk(codeA, codeB):
    print("Beginning RelativeRisk(codeA, codeB)")
    print(f'Primary diagnosis: {codeA}. Secondary diagnosis: {codeB}.')
    countTotal = len(icd9CountsPrim) + sum(icd9CountsSec.values())
    print(f'Count total: {countTotal}')
    countA = icd9CountsPrim[codeA]
    print(f'Count of primary diagnosis {codeA}: {countA}')
    countB = icd9CountsSec.get(codeB, 0)
    print(f'Count of secondary diagnosis {codeB}: {countB}')
    countAB = countA + countB
    print(f'count A + count B = {countAB}')
    relativeRisk = (countAB / countTotal) / ((countA / countTotal)*(countB / countTotal))
    print(f'Relative risk: {relativeRisk}')
    print("Ending RelativeRisk(codeA, codeB)")
    return relativeRisk

def ClusterRRTotal(pClusterSet, pClusterIndex):
    print("Beginning ClusterRRTotal(clusterIndex)")
    print(f'pClusterIndex: {pClusterIndex}')
    # Get the indices of the data points in the current cluster
    indices = np.where(pClusterSet.labels_ == pClusterIndex)[0]
    print(f'all indices for clusterIndex {clusterIndex} retrieved')
    clusterTotal = 0.0
    print("clusterTotal initialised")
    # Loop through all possible pairs of data points in the cluster
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            # Calculate the relative risk for the pair of data points
            relativeRisk = RelativeRisk(XPCA[indices[i]], XPCA[indices[j]])
            clusterTotal += relativeRisk
    print("Looped through all possible pairs of data points in the cluster and calculated relative risk")
    print(f'Final cluster total: {clusterTotal}')
    print("Ending ClusterRRTotal(clusterIndex)")
    return clusterTotal

def MergeClusters(pClusterSet, pClusterA, pClusterB, k):
    print("Beginning MergeClusters(pClusterSet, pClusterA, pClusterB, k)")
    print(f'pClusterA: {pClusterA}. pClusterB: {pClusterB}. k: {k}')
    # merge the two clusters into a new cluster with label 999
    newLabels = np.where(pClusterSet.labels_ == pClusterA, 999, pClusterSet.labels_)
    print("Cluster A's labels changed to cluster 999")
    newLabels = np.where(newLabels == pClusterB, 999, newLabels)
    print("Cluster B's labels changed to cluster 999")

    # delete the original clusters from the labels array
    newLabels = np.delete(newLabels, np.where(newLabels == pClusterA))
    newLabels = np.delete(newLabels, np.where(newLabels == pClusterB))
    print("Deleted clusters A and B from the labels")

    # create a new kmeans object with the updated labels
    newKmeans = KMeans(n_clusters=k-1, random_state=0).fit(XPCA)
    print("created new kmeans object")
    newKmeans.labels_ = newLabels
    print("assigned new labels to kmeans object")
    print("Finised MergeClusters(pClusterSet, pClusterA, pClusterB, k)")
    return newKMeans
            
def GrowCluster(pClusterSet, pClusterC, pUnbalanceFactor):
    print("Beginning GrowCluster(pClusterSet, pClusterC, pUnbalanceFactor)")
    print(f'pClusterC: {pClusterC}. pUnbalanceFactor: {pUnbalanceFactor}')
    #get clusterC data
    indicesC = np.where(pClusterSet.labels_ == pClusterC)[0]
    print("created list of data points for cluster C")
    
    #find out how many nodes should be in the new cluster (% of original cluster)
    targetSize = int(len(indicesC) * pUnbalanceFactor)
    print(f'Calculated target size: {targetSize})')

    # Select a random primary diagnosis data point from clusterC
    primIndices = np.where(XPCA[indicesC][:, 0] == 1)[0]
    print("created a list of all primary diagnoses present in the cluster")
    randomPrimIndex = np.random.choice(primaryIndices)
    print("selected a random primary diagnosis from the cluster")
    randomPrimLabel = pClusterSet.n_clusters + 1  # assign a new label not currently in kmeans
    print("select a new cluster label not currently present in the cluster set")
    pClusterSet.labels_[indices[randomPrimIndex]] = randomPrimLabel
    print(f'Data point {randomPrimIndex} assigned to cluster {randomPrimLabel}')

    # Create a new clusterD with the randomly selected primary diagnosis data point
    indicesD = np.array([indicesC[randomPrimIndex]])
    print("New cluster D created, containing primary diagnosis")
    indicesC = np.delete(indicesC, randomPrimIndex)
    print("Removed primary diagnosis from cluster C")

    # Loop through the secondary diagnosis data points and move them to clusterD if they have a significant relative risk
    while len(indicesD) < targetSize:
        relativeRisks = []
        for index in indicesC:
            relativeRisks.append(RelativeRisk(XPCA[indicesD[0]], XPCA[index]))
        print("Created list of relative risks for every data point in cluster C, in comparison to the primary diagnosis in cluster D")
        
        # Find the data point with the highest relative risk
        maxIndex = np.argmax(relativeRisks)
        print("Selected data point with largest relative risk")

        # If the relative risk is significant, move the data point to clusterD
        if relativeRisks[maxIndex] > 1:
            print("Relative risk is significant against threshold 1")
            indicesD = np.append(indicesD, indicesD[maxIndex])
            print("Moved data point with largest relative risk to cluster D")
            indicesC = np.delete(indicesC, maxIndex)
            print("Removed data point with largest relative risk from cluster C")
        else:
            print("Relative risk is not significant against threshold 1. Break.")
            break

    # Update kmeans labels
    pClusterSet.labels_[indicesC] = pClusterC
    pClusterSet.labels_[indicesD] = randomPrimLabel
    print("Updated cluster set labels for clusters C and D")
    
    # Update kmeans cluster centers
    pClusterSet.cluster_centers_ = np.vstack((pClusterSet.cluster_centers_, XPCA[indicesD[0]]))
    print("Updated cluster centers")
    
    # Update kmeans n_clusters and return
    pClusterSet.n_clusters += 1
    print("Updated number of clusters in set")

    print("Finished GrowCluster(pClusterSet, pClusterC, pUnbalanceFactor)")
    return pClusterSet


def SelectAB(pClusterSet):
    print("Beginning SelectAB(pClusterSet)")
    #pick a random cluster A
    clusterA = random.randint(0, pClusterSet.n_clusters-1)
    print(f'Picked random cluster A: {clusterA}')
    #pick a random cluster B
    clusterB = random.randint(0, pClusterSet.n_clusters-1)
    print(f'Picked random cluster B: {clusterB}')
    if clusterA == clusterB:
        clusterB = random.randint(0, pClusterSet.n_clusters-1)
        print(f'clusters A and B are the same. New cluster B: {clusterB}')
    threshold = 0.75
    print(f'Threshold set at {threshold}')
    #calculate sum of their weights
    clusterARR = ClusterRRTotal(pClusterSet, clusterA)
    print(f'Cluster ARR: {clusterARR}')
    clusterBRR = ClusterRRTotal(pClusterSet, clusterB)
    print(f'Cluster BRR: {clusterBRR}')
    probability = (clusterARR + clusterBRR) / E
    print(f'probability = {clusterARR} + {clusterBRR} / {E} = {probability}')
    #if this value divided by the total sum of weights for the whole cluster set is greater than t
    if probability >= threshold:
        print("Probability is greater than threshold")
        print("Finished SelectAB(pClusterSet)")
        return(clusterA, clusterB)
    else:
        print("Probability is not greater than threshold")
        SelectAB(pClusterSet)

def KAlgorithm(pClusterSet, k):
    print(f'Beginning KAlgorithm(pClusterSet, k). k = {k}')
    labels = pClusterSet.labels_
    print("List of labels created")
    deltaThreshold = 0
    print(f'Delta theshold initialised at {deltaThreshold}')

    while True:
        # Calculate relative risks for all pairs of points in different clusters
        relativeRisks = np.zeros((k, k))
        print("Initialised numpy array for relative risks")
        for i in range(k):
            for j in range(k):
                if i != j:
                    indicesI = np.where(labels == i)[0]
                    indicesJ = np.where(labels == j)[0]
                    print("For every i,j in every cluster, lists of indices i and j created")
                    for indexI in indicesI:
                        for indexJ in indicesJ:
                            relativeRisks[i, j] = max(relativeRisks[i, j], RelativeRisk(XPCA[indexI], XPCA[indexJ]))
                    print("Every relative risk calculated for pairs of i,j indices. Max value selected if calculation previously performed.")
       
                            
        # Find the point with the largest relative risk to any point in any other cluster
        maxDelta = 0
        maxI = -1
        maxJ = -1
        print("maxDelta, maxI, maxJ, initialised")
        for i in range(k):
            indicesI = np.where(labels == i)[0]
            print(f'all data points for cluster {i} calculated')
            for indexI in indicesI:
                for j in range(k):
                    if i != j:
                        indicesJ = np.where(labels == j)[0]
                        print(f'All data points for cluster {j} calculated')
                        for indexJ in indicesJ:
                            delta = RelativeRisk(XPCA[indexI], XPCA[indexJ]) - relativeRisks[i, j]
                            print(f'Relative risk for new cluster calculated, subtracted current relative risk. Delta = {delta}')
                            if delta > maxDelta:
                                print("Delta > maxDelta")
                                maxDelta = delta
                                maxI = i
                                maxJ = j
                                maxIndexI = indexI
                            else:
                                print("Delta < maxDelta. No action.")
                                
        # If the largest relative risk is above the threshold, move the point to the other cluster
        if maxDelta > deltaThreshold:
            labels[maxIndexI] = maxJ
            print('maxDelta > deltaThreshold. Move data point to new cluster')
        else:
            print('No changes made this iteration. Break loop')
            break

    # Create a new kmeans object with the updated labels
    newKmeans = KMeans(n_clusters=k, random_state=0).fit(XPCA)
    print("New kmeans object created")
    newKmeans.labels_ = labels
    print("Labels updated")
    print("Finished KAlgorithm(pClusterSet, k)")
    return newKmeans

def MAlgorithm(k):
    print(f'Beginning MAlgorithm(k). k = {k}')
    #create copy of prevaling cluster set as newClusterSet
    newClusterSet = clusterSet.copy()
    print("Copy of cluster set created")
    clusterA, clusterB = SelectAB(newClusterSet)
    print(f'Clusters {clusterA} and {clusterB} selected for merging')
    #with these IDs, merge the two clusters in the newClusterSet
    newClusterSet = MergeClusters(newClusterSet, clusterA, clusterB, k)
    print(f'Clusters {clusterA} and {clusterB} merged.')
    #randomly select another cluster for splitting, a cluster seed and unbalanceFactor
    clusterC = random.randint(0, newClusterSet.n_clusters-1)
    print(f'Cluster {clusterC} selected for splitting')
    #create a boolean array for all values in XPCA where they are in clusterC, then select from these points
    #clusterSeed = random.choice(XPCA[kmeans.labels_ == clusterC])
    unbalanceFactor = random.uniform(0.05, 0.95)
    print(f'Unbalance Factor {unbalanceFactor} selected')
    newClusterSet = GrowCluster(newClusterSet, clusterC, unbalanceFactor)
    print("New cluster grown")
    """
also in bisecting k means:
 Measure the distance for each intra cluster (SSE).
5. Select the cluster that have the largest SSE and
split it to 2 clusters using k-means.
"""

    newClusterSet = KAlgorithm(newClusterSet, k)
    print("K algorithm successfully run for this repeat")

    #calculate weights between data points in each cluster
    Enew = 0.0
    # Loop through each cluster
    for i in range(k):
        clusterTotal = ClusterRRTotal(newClusterSet, i)
        Enew += clusterTotal
    print(f'Total of relative risks for each cluster: {Enew}')

    if (Enew > E):
        print("New value of E is greater than current value. Update solution")
        E = Enew
        clusterSet = newClusterSet
    else:
        print("New value of E is less than current value. Discard solution.")

"""
    if (silhouetteScores2 == None) and (chScores2 == None) and (sseScores2 == None):
        silhouetteScores2.append(silhouetteAvg)
        chScores2.append(chIndex)    
        sseScores2.append(sse)
        clusterSet = newClusterSet
    elif (silhouetteAvg >= silhouetteScores2[-1]) and (chIndex >= chScores2[-1]) and (sse <= sseScores2[-1]):
        silhouetteScores2.append(silhouetteAvg)
        chScores2.append(chIndex)    
        sseScores2.append(sse)
        clusterSet = newClusterSet
"""
    
R = 10
print(f'R set to {R}')
clusterSet = None
E = 0.0
print("Cluster set and E variables initialised")

silhouetteScores2 = []
print("silhouette score list initialised")
chScores2 = []
print("calinski harabasz index list initialised")
sseScores2 = []
print("sse score list initialised")

for k in K:
    clusterSet = KMeans(n_clusters=k, n_init=10, random_state=0).fit(XPCA)
    print(f'Cluster set for k={k} created')
    for i in range(i, R):
        print(f'R = {R}')
        #perform MAlgorithm passing cluster set to it
        MAlgorithm(k)
        print("M Algorithm successfully performed")

    silhouetteAvg = silhouette_score(XPCA, newClusterSet.labels_)
    print(f'Silhouette Average: {silhouetteAvg}')
    chIndex = calinski_harabasz_score(XPCA, newClusterSet.labels_)
    print(f'Calinski Harabasz Score: {chIndex}')
    sse = newClusterSet.inertia_
    print(f'SSE Score: {sse}')
    
    plt.scatter(xtsne[:,0], xtsne[:,1], c=clusterSet.labels_, cmap='rainbow')
    plt.show()


