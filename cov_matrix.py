import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #for confusion matrix
#For data visualization
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go


import warnings 
warnings.filterwarnings('ignore')


data=pd.read_csv("data1.csv")

df=data.dropna(axis=0)
df.index=range(0,len(df),1)

## Clear ? 
df["Bare Nuclei"] = df["Bare Nuclei"][df["Bare Nuclei"]!='?']
df.dropna(inplace=True)
df["Bare Nuclei"] = df["Bare Nuclei"].astype("int64")

## delete "class" & "id"
x = df.drop(labels=["id"], axis=1)
f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(x.corr(),annot=True,fmt=".2f",ax=ax,linewidths=0.5,linecolor="orange")

# plot heatmap
plt.title('Covariance Matrix of Dataset')
plt.show()



