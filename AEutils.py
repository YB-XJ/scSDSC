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


def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

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

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    NMI = metrics.normalized_mutual_info_score(gt_s, c_x)
    #ARI=metrics.adjusted_rand_score(gt_s, c_x)
    purity = 0
    N = gt_s.shape[0]
    Label1 = np.unique(gt_s)
    nClass1 = len(Label1)
    Label2 = np.unique(c_x)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    for label in Label2:
        tempc = [i for i in range(N) if s[i] == label]
        hist,bin_edges = np.histogram(gt_s[tempc],Label1)
        purity += max([np.max(hist),len(tempc)-np.sum(hist)])
    purity /= N
    return missrate,NMI,purity

def display(Coef, subjs, d, alpha, ro,numc = None,label = None):
    if numc is None:
        numc = np.unique(subjs).shape[0]
    if label is None:
        label = subjs
    y_x, L = post_proC(Coef, numc, d, alpha, ro)
    NMI=np.round(metrics.normalized_mutual_info_score(label,y_x),4)
    ARI=np.round(metrics.adjusted_rand_score(label, y_x),4)
    #missrate_x, NMI, purity = err_rate(label, y_x)
    #acc_x = 1 - missrate_x
    #print("our accuracy: %.4f" % acc_x)
    #print("our NMI: %.4f" % NMI, "our purity: %.4f" % purity)
    print("NMI: ", NMI, 'ARI:',ARI)
    return (L,y_x)
    #return acc_x,L,y_x


def display1(Coef, subjs, d, alpha, ro,numc = None,label = None):
    if numc is None:
        numc = np.unique(subjs).shape[0]
    if label is None:
        label = subjs
    y_x, L = post_proC(Coef, numc, d, alpha, ro)
    y_x=best_map(label,y_x)
    return L,y_x

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
    
def contrastive_loss(y,d):
    tmp = y*tf.square(d)
    tmp2= (1-y)*tf.square(tf.maximum((1-d),0))
    return 0.5*tf.reduce_sum(tmp+tmp2)

