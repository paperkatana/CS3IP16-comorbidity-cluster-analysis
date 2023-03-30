import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score

# read in the data
df = pd.read_csv("C:/Users/HP/Desktop/Jupyter/Untitled Folder/ADMISSIONS_PRIMARY_SECONDARY.csv")

# extract relevant columns
df = df[['HADM_ID', 'PRIMARY_DIAGNOSIS', 'SECONDARY_DIAGNOSES']]

# convert secondary diagnoses to list
df['SECONDARY_DIAGNOSES'] = df['SECONDARY_DIAGNOSES'].str.split(',')

# convert ICD9 codes to CCS categories
icd9_to_ccs = pd.read_csv('icd9_ccs_mapping.csv')
df['PRIMARY_DIAGNOSIS_CCS'] = df['PRIMARY_DIAGNOSIS'].map(icd9_to_ccs.set_index('icd9')['ccs'])
df['SECONDARY_DIAGNOSES_CCS'] = df['SECONDARY_DIAGNOSES'].apply(lambda x: [icd9_to_ccs.loc[icd9_to_ccs['icd9'] == i]['ccs'].values[0] for i in x])

# convert secondary diagnoses to binary features
mlb = MultiLabelBinarizer()
df = df.join(pd.DataFrame(mlb.fit_transform(df['SECONDARY_DIAGNOSES']), columns=mlb.classes_, index=df.index))

# perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(df.drop(['HADM_ID', 'PRIMARY_DIAGNOSIS', 'SECONDARY_DIAGNOSES'], axis=1))

silhouette_avg = silhouette_score(df.drop(['HADM_ID', 'PRIMARY_DIAGNOSIS', 'SECONDARY_DIAGNOSES'], axis=1), kmeans.labels_)
print("The average silhouette_score is :", silhouette_avg)

# visualize results
# ...
