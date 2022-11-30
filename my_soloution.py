import pandas as pd
import numpy as np
df=pd.read_csv('DATASET.csv')

df=df.replace('_',np.nan)
df=df.replace('__',np.nan)
df=df.replace('_ ',np.nan)
df=df.replace(' _',np.nan)
df=df.replace(' ',np.nan)
df=df.replace('?',np.nan)
df['Cr']=pd.to_numeric(df['Cr'],errors='coerce')
df['Cr-1']=pd.to_numeric(df['Cr-1'],errors='coerce')
df['Cr-7']=pd.to_numeric(df['Cr-7'],errors='coerce')

#filling missing values with k nearest neigbors
k=5
from sklearn.impute import KNNImputer
imputer=KNNImputer(n_neighbors=k)
df_knn=df.copy()
for col in df_knn:
   X=df_knn[col].to_numpy().copy()
   X.resize((1,X.shape[0]))
   imputer.fit(X)
   Xtrans=imputer.transform(X).copy()
   Xtrans.resize((X.shape[1], ))
   df[col]=Xtrans
df=df.copy()
#classifying damage level 
df['class']=None
df.loc[(df['Cr-1']-df['Cr']<=0.3) |(df['Cr-1']/df['Cr']<1.5),'class']=0
df.loc[(df['Cr-7']-df['Cr']<=0.3) |(df['Cr-7']/df['Cr']<1.5),'class']=0

df.loc[(df['Cr-1']-df['Cr']>0.3) |((df['Cr-1']/df['Cr']>=1.5)&(df['Cr-1']/df['Cr']<2)),'class']=1
df.loc[(df['Cr-7']-df['Cr']>0.3) |((df['Cr-7']/df['Cr']>=1.5)&(df['Cr-7']/df['Cr']<2)),'class']=1

df.loc[(df['Cr-1']/df['Cr']>=2)&(df['Cr-1']/df['Cr']<3),'class']=2
df.loc[(df['Cr-7']/df['Cr']>=2)&(df['Cr-7']/df['Cr']<3),'class']=2

df.loc[(df['Cr-1']>4)|(df['Cr-1']/df['Cr']>=3),'class']=3
df.loc[(df['Cr-7']>4)|(df['Cr-7']/df['Cr']>=3),'class']=3
print(df.isna().any())
