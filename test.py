import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.externals import joblib


kmeans = joblib.load('model.pkl')
df = pd.read_csv("df.csv", sep="\t")
df_original = pd.read_csv("df_original.csv", sep="\t")
df_test= pd.read_csv("df_test.csv", sep="\t")
labels = kmeans.labels_

df_original['label']=pd.Series(labels)

#print(df_original.groupby('label').size())


df.drop('Unnamed: 0', axis=1, inplace=True)
df_test.drop('Unnamed: 0', axis=1, inplace=True)
df_original.drop('Unnamed: 0', axis=1, inplace=True)
#print(df_test)
print(kmeans.predict(df_test))

# g1 =df_original[df_original.label== 1]
# g2 =df_original[df_original.label== 2]
# g3 =df_original[df_original.label== 3]
# g4 =df_original[df_original.label== 4]
# g5 =df_original[df_original.label== 5]
# g6 =df_original[df_original.label== 6]
# g7 =df_original[df_original.label== 7]
# g8 =df_original[df_original.label== 8]
# g9 =df_original[df_original.label== 9]
# g0 =df_original[df_original.label== 0]
#print(df)

#g1.hist()
#g2.hist()
#plt.show()