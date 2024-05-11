from Pfeature.pfeature import paac
from Pfeature.pfeature import apaac
import pandas as pd
import numpy as np
paac("protein.seq",1,"paac_umami.csv")
# apaac("protein.seq","apaac_umami.csv")

df_data=pd.read_csv("merged_data_modified.csv")
df_data.head()
df_protein=pd.read_csv("paac_umami.csv")
df_protein.head()

df_final=pd.concat([df_data, df_protein], axis=1)

df_final.drop(['seq'], axis=1, inplace=True)
df_final.head()



df_final.to_csv('paac_data.csv', index=False)