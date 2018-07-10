import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.externals import joblib
#from sklearn.metrics import pairwise_distances_argmin_min
#import tensorflow as tf
#from tensorflow.python.data import Dataset

#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

df = pd.read_csv("DataServices_Datastream_Raw.csv", sep=",")
#print(df.head())

# delete columns not used
# dwell_type, urban_rural_flag, lenght_of_recidence, 
df.drop({'id', 'pres_kids', 'homeowner', 'metro_nonmetro', 'num_adults_range', 'length_of_residence_range'}, axis=1, inplace=True)
#print(df)



# change columns with text for number
# gender
mask = df.gender == 'M'
df.loc[mask, 'gender'] = 1
mask = df.gender == 'F'
df.loc[mask, 'gender'] = 0
#print(df.gender)

# hh_marital_status
mask = df.hh_marital_status == 'M'
df.loc[mask, 'hh_marital_status'] = 0
mask = df.hh_marital_status == 'S'
df.loc[mask, 'hh_marital_status'] = 1
#print(df.hh_marital_status)

# dwell_type
mask = df.dwell_type == 'P'
df.loc[mask, 'dwell_type'] = 0
mask = df.dwell_type == 'S'
df.loc[mask, 'dwell_type'] = 1
mask = df.dwell_type == 'A'
df.loc[mask, 'dwell_type'] = 2
mask = df.dwell_type == 'M'
df.loc[mask, 'dwell_type'] = 3
#print(df.dwell_type)

#urban_rural_flag
mask = df.urban_rural_flag == 'Rural'
df.loc[mask, 'urban_rural_flag'] = 0
mask = df.urban_rural_flag == 'Urban'
df.loc[mask, 'urban_rural_flag'] = 1

#age_range
df.age_range = df.age_range.apply(lambda val: ( int(val.split('-')[1]) + int(val.split('-')[0]) ) / 2)


# clean data null

#length_of_residence
df.length_of_residence = df.length_of_residence.fillna( int(df.length_of_residence.mean()) )

#gender
df.gender = df.gender.fillna(0)

#num_adults 
df.num_adults = df.num_adults.fillna( int(df.num_adults.mean()) )

#hh_marital_status 
#print(df.hh_marital_status.mean())
df.hh_marital_status = df.hh_marital_status.fillna( 0 )

#dwell_type 
#print(df.dwell_type.mean())
df.dwell_type = df.dwell_type.fillna(method='ffill')

#urban_rural_flag 
#print(df.urban_rural_flag.mean())
df.urban_rural_flag = df.urban_rural_flag.fillna( 1 )

#age_range
#print(df.age_range.mean())
df.age_range = df.age_range.fillna( int(df.age_range.mean()) )

print('Data cleaned \n \n')
#print(df)

df_original=df

#normalize data
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

X_new=df[70002:len(df)]
df=df[1:70001]
#print(df)

#print('df size')
#print(df.shape)

#K-Means Agrupamiento

kmeans = KMeans(n_clusters=10).fit(df.values)
centroids = kmeans.cluster_centers_

joblib.dump(kmeans, 'model.pkl') 
 
df_original.to_csv('df_original.csv',sep='\t')
df.to_csv('df.csv',sep='\t')
X_new.to_csv('df_test.csv',sep='\t')
#model_loaded = joblib.load('model.pkl')

#print('centroids')
#print(centroids)


#print('grupos')
#print(labels)

#print(df_original)
#print('len')
#print (labels.shape)


# test_label=kmeans.predict(X_new.values)
# print(test_label)

# result=pd.DataFrame(test_label)
# print(result.head())


#print(df.groupby('label').size())



# g1 = df.iloc[df.label==1]
# print(g1.hist())
#kmeans.predict(X_new) #Para evaluar un nuevo dato