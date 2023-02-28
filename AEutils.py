import numpy as np
from munkres import Munkres
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize,MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import neighbors
from PIL import Image
import tensorflow as tf
import seaborn as sns
#from mpl_toolkits.mplot3d import Axes3D


def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp

def post_proC(C, K, d, alpha, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    #C = thrC(C,ro)
    # drawC(C)
    C = 0.5*(C + C.T)
    r = min(d*K + 1, C.shape[0]-1)
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize',random_state=22)
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def display(Coef, subjs, d, alpha, ro,numc = None,label = None):
    if numc is None:
        numc = np.unique(subjs).shape[0]
    if label is None:
        label = subjs
    y_x, L = post_proC(Coef, numc, d, alpha, ro)
    NMI=np.round(metrics.normalized_mutual_info_score(label,y_x),3)
    ARI=np.round(metrics.adjusted_rand_score(label, y_x),3)
    print("NMI: ", NMI, 'ARI:',ARI)
    return (L,y_x)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orangered','greenyellow','darkolivegreen','maroon','darkgreen','darkslateblue','deeppink','goldenrod','teal','cornflowerblue','darksalmon','lightcoral','fuchsia']
marks = ['o','+','.']
def visualize(Img,Label,AE=None,filep=None):
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax1 = fig.add_subplot(111)
    n = Img.shape[0]
    if AE is not None:
        bs = AE.batch_size
        Z = AE.transform(Img[:bs,:])
        Z = np.zeros([Img.shape[0], Z.shape[1]])
        for i in range(Z.shape[0] // bs):
            Z[i * bs:(i + 1) * bs, :] = AE.transform(Img[i * bs:(i + 1) * bs, :])
        if Z.shape[0] % bs > 0:
            Z[-bs:, :] = AE.transform(Img[-bs:, :])
    else:
        Z = Img
    Z_emb = TSNE(n_components=2).fit_transform(Z, Label)
    # print(Z_emb)
    lbs = np.unique(Label)#lbs：0-9
    for ii in range(lbs.size):
        Z_embi = Z_emb[[i for i in range(n) if Label[i] == lbs[ii]]].transpose()#shape:2*500
        # print(Z_embi)
        ax1.scatter(Z_embi[0], Z_embi[1], color=colors[ii], marker=marks[ii // 10], label=str(ii),s=3)
    ax1.legend()
    if filep is not None:
        plt.savefig(filep)
    plt.show()

def drawC(C,name='C-L2.png',norm=False):
    C = np.abs(C)
    C = C * (np.ones_like(C)-np.eye(C.shape[0]))
    if norm:
        C = C / np.sum(C,axis=1,keepdims=True)
    min_max_scaler = MinMaxScaler(feature_range=[0,255])
    CN = min_max_scaler.fit_transform(C)
    CN = CN + 255*np.eye(C.shape[0])
    IC = Image.fromarray(CN).convert('L')
    sns.heatmap(CN)
    plt.show()
    IC.save(name)
    # IC.show()
    
