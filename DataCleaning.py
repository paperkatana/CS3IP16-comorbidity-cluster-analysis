import pandas as pd
import re

#import the patients CSV file into a DataFrame and check the contents
patients = pd.read_csv('./mimic-iii/PATIENTS.csv')
print(len(patients.index))
print(patients.head())

#import the diagnoses CSV file into a DataFrame and check the contents
diagnoses = pd.read_csv('./mimic-iii/DIAGNOSES_ICD.csv')
print(len(diagnoses.index))
print(diagnoses.head())

#remove all ICD 9 codes that aren't exclusively numeric
diagnoses['ICD9_CODE'] = diagnoses['ICD9_CODE'].astype(str)
diagnoses = diagnoses[diagnoses['ICD9_CODE'].apply(lambda x: re.match(r'^\d+$', x) is not None)]
print(len(diagnoses.index))
print(diagnoses.head())

# filter out admission_ids without a primary diagnosis
diagnoses = diagnoses.groupby('HADM_ID').filter(lambda x: (x['SEQ_NUM'] == 1.0).any())
print(len(diagnoses.index))
print(diagnoses.head())

diagnosesPrimary = diagnoses[diagnoses['SEQ_NUM'] == 1.0].copy()
diagnosesSec = diagnoses[diagnoses['SEQ_NUM'] != 1.0].copy()
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

patients.drop(columns=['ROW_ID', 'GENDER', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN'], inplace=True)
print(patients.head())

outputData = pd.merge(patients, diagnosesPrimary, on='SUBJECT_ID')
outputData.drop_duplicates(inplace=True)

outputData = pd.merge(outputData, diagnosesSec, on='HADM_ID')
outputData.drop_duplicates(inplace=True)

outputData.rename(columns={'ICD9_CODE_x':'PRIMARY_DIAGNOSIS', 'ICD9_CODE_y':'SECONDARY_DIAGNOSES', 'SUBJECT_ID_x':'SUBJECT_ID'}, inplace=True)
outputData.drop(columns=['SUBJECT_ID_y'], inplace=True)
outputData.drop_duplicates(inplace=True)
print(outputData.head())

outputData.to_csv('ADMISSIONS_PRIMARY_SECONDARY.csv')

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
diagData.to_csv('DATA.csv')
