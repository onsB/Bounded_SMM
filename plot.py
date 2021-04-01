import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


def plot_tsne(data, clusters, fig, color_list = ['blue','red','green']):
    '''
    plots the data with colormap following cluster values using TSNE
    data: pandas df
    clusters: np array of cluster values given by BSMM
    '''
    cluster_found_sr = pd.Series(clusters, name='cluster')
    data = data.set_index(cluster_found_sr, append = True)
    res = TSNE().fit_transform(np.asarray(data))
    cluster_values = sorted(data.index.get_level_values('cluster').unique())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cluster_values, color_list)
    axarr = fig.add_subplot(1,1,1)
    plt.scatter(res[:, 0], res[:, 1], c = data.index.get_level_values('cluster'),
                cmap=cmap, alpha=0.6,)

    # plt.show()
    plt.title('clusters')
    return fig

def plot_timeseries(data, clusters,fig, color_list = ['blue','red','green']):
    cluster_found_sr = pd.Series(clusters, name='cluster')
    data = data.set_index(cluster_found_sr, append=True)
    cluster_values = sorted(data.index.get_level_values('cluster').unique())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cluster_values, color_list)
    axarr = fig.add_subplot(1, 1, 1)
    data.T.plot(figsize=(13, 8), cmap=cmap, legend=True, alpha=0.02)
    return fig

def plot_separate_clusters(data, clusters, fig):
    nb_clust = np.unique(clusters).shape[0]
    data_ = data.copy()
    try:
        data_['cluster'] = clusters
    except:
        raise ValueError('clusters parameter must be a np array of the same size as data.shape[0]')
    for i in range(nb_clust):
        fig.add_subplot(nb_clust,1,i+1)
        current = data_[data_['cluster']==i].drop(columns=['cluster'])
        ttl = 'time-series of cluster number '+str(i)
        current.T.plot(figsize = (13,8), title=ttl, color = 'blue')
    return fig
