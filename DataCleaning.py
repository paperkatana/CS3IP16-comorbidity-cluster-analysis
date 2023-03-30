import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer

#import the patients CSV file into a DataFrame and check the contents
patients = pd.read_csv('./data/PATIENTS.csv')
print(len(patients.index))
print(patients.head())

#import the diagnoses CSV file into a DataFrame and check the contents
diagnoses = pd.read_csv('./data/DIAGNOSES_ICD.csv')
print(len(diagnoses.index))
print(diagnoses.head())

#remove all ICD 9 codes that aren't exclusively numeric
diagnoses['ICD9_CODE'] = diagnoses['ICD9_CODE'].astype(str)
diagnoses = diagnoses[diagnoses['ICD9_CODE'].apply(lambda x: re.match(r'^\d+$', x) is not None)]
print(len(diagnoses.index))
print(diagnoses.head())

#import the diagnoses CSV file into a DataFrame and check the contents
admissions = pd.read_csv('./ADMISSIONS.csv')
print(len(admissions.index))
print(admissions.head())

diagnosesPrimary = pd.DataFrame({'ROW_ID': pd.Series(dtype='int'),
                    'SUBJECT_ID': pd.Series(dtype='int'),
                    'HADM_ID': pd.Series(dtype='int'),
                    'SEQ_NUM': pd.Series(dtype='float'),
                    'ICD9_CODE': pd.Series(dtype='object')})

diagnosesSec = pd.DataFrame({'ROW_ID': pd.Series(dtype='int'),
                    'SUBJECT_ID': pd.Series(dtype='int'),
                    'HADM_ID': pd.Series(dtype='int'),
                    'SEQ_NUM': pd.Series(dtype='float'),
                    'ICD9_CODE': pd.Series(dtype='object')})

#for each row in diagnoses table, if seqno=1, append that row to the diagnosesPrimary table
for index, row in diagnoses.iterrows():
    #currentRow = diagnoses.iloc[i,0]
    #print("Current row: ", diagnoses.iloc[i,0])
    bottomRowP = len(diagnosesPrimary.index) #pointer to last row in diagnosesPrimary
    bottomRowS = len(diagnosesSec.index)
    if(diagnoses.iloc[index,3] == 1.0):
        #diagnosesPrimary.append(diagnoses.loc[i,:])
        bottomRowP += 1  #new row 
        diagnosesPrimary.loc[bottomRowP] = diagnoses.loc[index] #add the row to the bottom of the df
    else:
        bottomRowS += 1  #new row 
        diagnosesSec.loc[bottomRowS] = diagnoses.loc[index] #add the row to the bottom of the df
    print(index, "finished.")
print(diagnosesPrimary.head())
print(diagnosesSec.head())

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

#create a copy of the patients table and keep only the subject id and expire flag
admCopy = admissions.copy()
admCopy.drop(columns=['ROW_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ADMISSION_TYPE', 
                           'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 
                           'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'EDREGTIME', 
                           'EDOUTTIME', 'DIAGNOSIS', 'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA'], inplace=True)
print(admCopy.head())

#combine the primary diagnoses and admission tables
outputData = pd.merge(admCopy, diagnosesPrimary, on='SUBJECT_ID')
outputData.drop_duplicates(inplace=True)
outputData.rename(columns={'ICD9_CODE':'PRIMARY_DIAGNOSIS', 'HADM_ID_x':'HADM_ID'}, inplace=True)
outputData.drop(columns=['HADM_ID_y'], inplace=True)
print(outputData.head())

#combine the secondary diagnosis and output tables
outputData = pd.merge(outputData, diagnosesSec, on='SUBJECT_ID')
outputData.drop_duplicates(inplace=True)
outputData.rename(columns={'ICD9_CODE':'SECONDARY_DIAGNOSES', 'HADM_ID_x':'HADM_ID'}, inplace=True)
outputData.drop(columns=['HADM_ID_y'], inplace=True)
print(outputData.head())

mlb = MultiLabelBinarizer()

outputData = outputData.join(pd.DataFrame(mlb.fit_transform(outputData['SECONDARY_DIAGNOSES']), columns=mlb.classes_, index=outputData.index
print(outputData.head())

outputData.to_csv('ADMISSIONS_PRIMARY_SECONDARY.csv')

