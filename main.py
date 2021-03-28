from BSMM_New import *
from mml import *
from pathlib import Path
# import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

def print_performance(posterior, target_labels):
    print(posterior.argmax(axis=1))
    print('%' + str(accuracy_score(posterior.argmax(axis=1), target_labels) * 100))
    return posterior.argmax(axis=1)


def main():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    n_comp = np.unique(labels).shape[0]
    # bsmm = BSMM(X=data, n_components=n_comp)
    # bsmm._init()
    # print('prior')
    # print(bsmm.prior)
    # print('sums up to %d' % bsmm.prior.sum())
    # print('up:')
    # print(bsmm.up)
    # print('low: ')
    # print(bsmm.low)
    # print('means:')
    # print(bsmm.means)
    # print('cov:')
    # print(bsmm.cov)

    # logl = bsmm.fit(tol= 1e-10)
    # print('----model selection-----')
    # l = [1,2,3,4,5,6]
    # optimal = model_selection(data,l)
    # print('best number of components: ',optimal)

    # print('after modifications')
    # print('prior')
    # print(bsmm.prior)
    # print('sums up to %d' % bsmm.prior.sum())
    # print('up:')
    # print(bsmm.up)
    # print('low: ')
    # print(bsmm.low)
    # print('means:')
    # print(bsmm.means)
    # print('cov:')
    # print(bsmm.cov)
    #h = bsmm.h(data,1)
    #print('h shape: ', h.shape)
    #print(h[:5])
    # z, resp = bsmm.E_step()
    # print('z matrix')
    # print(z[:3])
    # print('resp matrix')
    # print(resp[:3])
    # print('*** Results ***')
    # print_performance(bsmm.z,labels)
    # print('components: ',bsmm.n_components)

    #sns.heatmap(bsmm.cov[1], xticklabels=False, yticklabels=False)
    #plt.show()

    # heart = pd.read_csv('datasets/heart_dataset/heart_reduced.csv')
    # data2 = heart[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].to_numpy()
    # scaler = StandardScaler()
    # data2 = scaler.fit_transform(data2)
    # labels2 = heart['target'].to_numpy()
    #
    # n_comp2 = np.unique(labels2).shape[0]
    # bsmm2 = BSMM(X=data2, n_components=n_comp2)
    # bsmm2._init()
    # print('**second dataset**')
    # print('prior ')
    # print(bsmm2.prior)
    # print('sums up to %d'% bsmm2.prior.sum())
    # print('up:')
    # print(bsmm2.up)
    # print('low: ')
    # print(bsmm2.low)
    #print('means:')
    #print(bsmm2.means)
    #print('cov:')
    #print(bsmm2.cov)
    # bsmm2.fit()
    # print('after modifications')
    #print('prior')
    #print(bsmm2.prior)
    # print('sums up to %d' % bsmm2.prior.sum())
    # print('up:')
    # print(bsmm2.up)
    # print('low: ')
    # print(bsmm2.low)
    # print('means:')
    # print(bsmm2.means)
    # print('cov:')
    # print(bsmm2.cov)
    # print('*** Results ***')
    # print_performance(bsmm2.z,labels2)
    # print('components: ',bsmm2.n_components)

    p = Path(__file__).parent.parent

    # ---------household Power consumption dataset------------
    # ahu = pd.read_csv(
    #    'C:/Users/Admin/Desktop/concordia/thesis/datasets/HVAC/hvac_ahu_sensors/hvac_ahu_sensors/ahu1_evac.csv')
    # cons = pd.read_csv('C:/Users/Admin/Desktop/concordia/thesis/datasets/smart buildings/household data consumption/ready_data.csv')
    # cons = pd.read_csv('C:/Users/OnsB/Desktop/thesis/datasets/smart buildings/household data consumption/ready_data.csv')
    cons = pd.read_csv('datasets/household_consumption.csv')
    # the data is already scaled with MinMax scaling, now we apply multidimensional scaling
    mds = MDS(10, random_state=0)
    cons = mds.fit_transform(cons)
    bsmm_cons = BSMM(X=cons, n_components=2)
    bsmm_cons._init()
    logl = bsmm_cons.fit(tol=1e-10)




if __name__ == "__main__":
    main()
