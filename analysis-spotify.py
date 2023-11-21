from __future__ import division, print_function, unicode_literals
import warnings
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings(action="ignore", message="^internal gelsd")
from sklearn.metrics import silhouette_score

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import seaborn as sns
#set style of plots
sns.set_style('white')

#define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)
sns.palplot(customPalette)

# Import my spotify data
songs = pd.read_csv("./data/songs_normalize.csv")
print(songs.info())

feature_names = ["danceability", "energy", "loudness", 
                 "speechiness", "acousticness", "instrumentalness",
                 "valence", "tempo"]
drop_feature_names = ["artist", "song", "duration_ms", "explicit", 
                      "year", "popularity", "genre", "mode", "key", "liveness"]

songs_feature = songs.copy()
songs_feature = songs_feature.drop(drop_feature_names, axis=1)
songs_feature.dropna(inplace=True)
print(songs_feature.head())
print(songs_feature.describe())

print("")
print("standard scaler")
std_song_feature = songs_feature.copy()
std = preprocessing.StandardScaler()
std_song_feature = std.fit_transform(songs_feature)
std_song_feature = pd.DataFrame(std_song_feature, columns=songs_feature.columns, index = songs_feature.index)
print(std_song_feature.head())
print(std_song_feature.describe())

X = std_song_feature.values


# # analysis elbow method
# sum_of_squared_distances = []
# K = range(1, 15)

# for k in K:
#     km = KMeans(n_clusters=k,
#                 init="k-means++",
#                 n_init=10,
#                 max_iter=1000,
#                 random_state=0)
#     km = km.fit(X)
#     sum_of_squared_distances.append(km.inertia_)

# fig, ax = plt.subplots()

# ax.plot(K, sum_of_squared_distances, marker="o")
# ax.set_xlabel("k")
# ax.set_ylabel("sum of squared distances")
# ax.set_title("Ellbow Method For Optomal K")

# print("get_xlabel()", ax.get_xlabel())  # get_xlabel() height (cm)
# print("get_ylabel()", ax.get_ylabel())  # get_ylabel() frequency
# print("get_title()", ax.get_title()) 
# plt.show()

#result K = 4
km = KMeans(n_clusters=4,
            init="k-means++",
            n_init=10,
            max_iter=1000,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

tsne = TSNE(n_components=2, perplexity=50)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
pc = pd.DataFrame(principal_components)
pc["label"] = y_km
pc.columns = ["x", "y", "label"]

cluster = sns.lmplot(data=pc, x='x', y='y', hue='label', 
                     fit_reg=False, legend=True, legend_out=True)
plt.show()

songs["label"] = y_km

songs = songs.sample(frac=1)
print(f"label count: {songs['label'].value_counts()}")
print("")
print(f"songs label 0: {songs[songs['label'] == 0].tail(10)}")

print("")
print(f"songs label 1: {songs[songs['label'] == 1].tail(10)}")

print("")
print(f"songs label 2: {songs[songs['label'] == 2].tail(10)}")

print("")
print(f"songs label 3: {songs[songs['label'] == 3].tail(10)}")