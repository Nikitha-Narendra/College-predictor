# program to clean and evenly distribute data

import pandas as pd
import numpy as np
import random

#read dataset
df = pd.read_csv('college_admission_dataset.csv')

# group and view the number of each outcome
colleges = df.groupby('college').size()
colleges.sort_values(inplace=True)
print(colleges)

#remove outliers
outliers = [col for col in colleges.index if colleges[col]<20]
df = df[~(df.college.isin(outliers))]

#the datase set is uneven
#reduce the frequency of each outcome to 30

outliers_30 = [col for col in colleges.index if colleges[col] >30]
df2 = df[df.college.isin(outliers_30)]
df = df[~(df.college.isin(outliers_30))]

df3 = df2[0:0]
df_extra = df2[0:0]

for outs in outliers_30:
    np.random.seed(10)
    df_extra = df2[df2['college'] == outs]
    remove_n = colleges[outs] - 30
    drop_indices = np.random.choice(df_extra.index, remove_n, replace=False)
    df_subset = df_extra.drop(drop_indices)
    df3 = df3.append(df_subset,ignore_index=True)

df = df.append(df3,ignore_index=True)

#check for duplicates
duplicateRowsDF = df[df.duplicated(['cet rank'])]
while not(duplicateRowsDF.empty):
    for i,row in duplicateRowsDF.iterrows():
        df.at[i, 'cet rank'] = df.at[i, 'cet rank'] + random.randint(-10,10)

    duplicateRowsDF = df[df.duplicated(['cet rank'])]

#covert categorial data(branch) into numerical
branch = pd.get_dummies(df['branch'],drop_first=True)
college = df['college']
df.drop(['college','branch'],axis=1,inplace=True)
df = pd.concat([df,branch,college],axis=1)

#store cleaned data
df.to_csv('cleaned_data_20.csv')
