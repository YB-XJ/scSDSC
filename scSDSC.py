import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import h5py
from sklearn import preprocessing
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from sklearn.manifold import TSNE
from sklearn import cluster
import umap.umap_ as umap


def display(Coef, subjs, d, alpha, ro,numc = None,label = None):
    if numc is None:
        numc = np.unique(subjs).shape[0]
    if label is None:
        label = subjs
    y_x, L = post_proC(Coef, numc, d, alpha, ro)
    NMI=np.round(metrics.normalized_mutual_info_score(label,y_x),4)
    ARI=np.round(metrics.adjusted_rand_score(label, y_x),4)
    print("NMI: ", NMI, 'ARI:',ARI)
    return (L,y_x)

def post_proC(C, K, d, alpha, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
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

#colors = plt.cm.rainbow(np.linspace(0, 1,20))
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
    #Z_emb = TSNE(n_components=2).fit_transform(Z, Label)
    Z_emb = umap.UMAP(n_components=2).fit_transform(Z, Label)
    lbs = np.unique(Label)#lbs：0-9
    for ii in range(lbs.size):
        Z_embi = Z_emb[[i for i in range(n) if Label[i] == lbs[ii]]].transpose()#shape:2*500
        # print(Z_embi)
        ax1.scatter(Z_embi[0], Z_embi[1], color=colors[ii], marker=marks[ii // 10], label=str(ii),s=3)
    ax1.legend()
    if filep is not None:
        plt.savefig(filep)
    plt.show()

class AE(object):
    def __init__(self, n_input, n_hidden, reg_constant1=1.0, re_constant2=1.0, re_constant3=1.0,re_constant4=1.0,
                 batch_size=200, reg=None,ds=None, \
                 noise=False, model_path=None, restore_path=None, \
                 logs_path='./logs'):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        usereg = 2
        # input required to be fed
        
        #self.x = tf.placeholder(tf.float32, [None, n_input[0]*n_input[1]])
        self.x = tf.placeholder(tf.float32, [None, n_input[1]])
        
        self.learning_rate = tf.placeholder(tf.float32, [])
        c_dim = batch_size * batch_size
        weights = self._initialize_weights()

        x_input = self.x
        latent = tf.layers.dense(x_input,n_hidden[0],activation=tf.nn.relu)
        latent = tf.layers.dense(latent, n_hidden[1],activation=tf.nn.relu)
        latent = tf.layers.dense(latent, n_hidden[2],activation=tf.nn.relu)
        latent = tf.layers.dense(latent, n_hidden[3], activation=None)
        
        z = latent
        if ds is not None:
            pslb = tf.layers.dense(z,ds,kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.softmax,name = 'ss_d')
            cluster_assignment = tf.argmax(pslb, -1)
            eq = tf.to_float(tf.equal(cluster_assignment,tf.transpose(cluster_assignment)))

        Coef = weights['Coef']
        #Coef=tf.reshape(latent,[batch_size, -1])
        z_c = tf.matmul(Coef, z)
        self.Coef = Coef
        self.z = z

        # self.x_r = self.decoder(latent_c, weights, shape)
        # self.x_r2 = self.decoder(latent, weights, shape)
        self.x_r = tf.layers.dense(z_c, n_hidden[2],activation=tf.nn.relu,name='d1')
        self.x_r = tf.layers.dense(self.x_r, n_hidden[1],activation=tf.nn.relu,name='d2')
        self.x_r = tf.layers.dense(self.x_r, n_hidden[0],activation=tf.nn.relu,name='d3')
        self.x_r = tf.layers.dense(self.x_r, n_input[1], activation=None, name='d4')

        self.x_r2 = tf.layers.dense(z, n_hidden[2], activation=tf.nn.relu,name='d1',reuse=True)
        self.x_r2 = tf.layers.dense(self.x_r2, n_hidden[1], activation=tf.nn.relu,name='d2',reuse=True)
        self.x_r2 = tf.layers.dense(self.x_r2, n_hidden[0], activation=tf.nn.relu,name='d3',reuse=True)
        #X_pre_recons=self.x_r2
        self.x_r2 = tf.layers.dense(self.x_r2, n_input[1], activation=None, name='d4', reuse=True)

        # self.x_r = self.x_r2
        # l_2 reconstruction loss
        self.reconst_cost = tf.reduce_sum(tf.square(tf.subtract(self.x_r, self.x)))
        self.reconst_cost_pre = tf.reduce_sum(tf.square(tf.subtract(self.x_r2, self.x)))
        tf.summary.scalar("recons_loss", self.reconst_cost)

        if usereg == 2: 
            self.reg_losses = tf.reduce_sum(tf.square(self.Coef))+tf.trace(tf.square(self.Coef))
        else:
            self.reg_losses = tf.reduce_sum(tf.abs(self.Coef))+tf.trace(tf.abs(self.Coef))

        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.square(tf.subtract(z_c, z)))

        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses)

        x_flattten = tf.reshape(x_input, [batch_size, -1])
        x_flattten2 = tf.reshape(self.x_r, [batch_size, -1])
        XZ = tf.matmul(Coef, x_flattten)
        self.selfexpress_losses2 = 0.5 * tf.reduce_sum(tf.square(tf.subtract(XZ, x_flattten)))

        normL = True
        absC = tf.abs(Coef)
        C = (absC + tf.transpose(absC)) * 0.5# * (tf.ones([Coef.shape[0].value,Coef.shape[0].value])-tf.eye(Coef.shape[0].value))
        C = C + tf.eye(Coef.shape[0].value)

        if normL == True:
            D = tf.diag(1.0 / tf.reduce_sum(C,axis=1))
            I = tf.eye(D.shape[0].value)
            L = I - tf.matmul(D,C)
            D = I
        else:
            D = tf.diag(tf.reduce_sum(C, axis=1))
            L = D - C
        XLX2 = tf.matmul(tf.matmul(tf.transpose(x_flattten),L),x_flattten2)

        XX = x_flattten - x_flattten2

        self.tracelossx =tf.reduce_sum(tf.square(XX)) +  2.0 * tf.trace(XLX2)#/self.batch_size
        self.d = tf.reduce_sum(C,axis=1)
        self.l = tf.trace(XLX2)


        regass = tf.to_float(tf.reduce_sum(pslb,axis=0))

        onesl=np.ones(batch_size)
        zerosl=np.zeros(batch_size)
        weight_label=tf.where(tf.reduce_max(pslb,axis=1)>0.8,onesl,zerosl)
        
        
        cluster_assignment1=tf.one_hot(cluster_assignment,ds)
        #label1=tf.one_hot(self.label,ds)
        self.w_weight=weight_label
        self.labelloss=tf.losses.softmax_cross_entropy(onehot_labels=cluster_assignment1,logits=pslb,weights=weight_label)
        self.graphloss = tf.reduce_sum(tf.nn.relu((1 - eq) * C) + tf.nn.relu(eq * (0.001 - C))) + tf.reduce_sum(tf.square(regass))
        self.loss = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses

        self.loss2 = ( self.reconst_cost+re_constant3 * self.tracelossx + +re_constant2 * self.selfexpress_losses +  +re_constant3 * self.labelloss+re_constant4 * self.graphloss)
        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss2)  
        # GradientDescentOptimizer #AdamOptimizer
        self.optimizer = self.optimizer2
        # GradientDescentOptimizer #AdamOptimizer
        self.optimizer_pre = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.reconst_cost_pre)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])
        # [v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden)
        
        all_weights['Coef'] = tf.Variable(
            # 1 * tf.eye(self.batch_size, dtype=tf.float32), name='Coef')
            1.0e-5 * (tf.ones([self.batch_size, self.batch_size], dtype=tf.float32)), name='Coef')

        return all_weights

    # Building the encoder
    def encoder(self, x, weights):
        shapes = []
        shapes.append(x.get_shape().as_list())
        layeri = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1, 2, 2, 1], padding='SAME'),
                                weights['enc_b0'])
        layeri = tf.nn.relu(layeri)
        shapes.append(layeri.get_shape().as_list())

        n_layers = len(self.n_hidden)
        iter_i = 1
        while iter_i < n_layers:
            layeri = tf.nn.bias_add(
                tf.nn.conv2d(layeri, weights['enc_w' + str(iter_i)], strides=[1, 2, 2, 1], padding='SAME'),
                weights['enc_b' + str(iter_i)])
            layeri = tf.nn.relu(layeri)
            shapes.append(layeri.get_shape().as_list())
            iter_i = iter_i + 1
        layer3 = layeri
        return layer3, shapes

    # Building the decoder
    def decoder(self, z, weights, shapes):
        n_layers = len(self.n_hidden)
        layer3 = z
        iter_i = 0
        while iter_i < n_layers:
            shape_de = shapes[n_layers - iter_i - 1]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack(
                [tf.shape(self.x)[0], shape_de[1], shape_de[2], shape_de[3]]), strides=[1, 2, 2, 1], padding='SAME'),
                            weights['dec_b' + str(iter_i)])
            layer3 = tf.nn.relu(layer3)
            iter_i = iter_i + 1
        return layer3

    def partial_fit(self, X, lr, noise=True,mode=0):  #
        if noise==True:
            X=X+0.15*np.random.randn(*X.shape)
        cost0, cost1, cost2,cost3, summary, _, Coef,d,l = self.sess.run((self.reconst_cost, self.selfexpress_losses,
                            self.selfexpress_losses2,self.tracelossx, self.merged_summary_op,
                            self.optimizer,self.Coef,self.d,self.l),feed_dict={self.x: X, self.learning_rate: lr})  #
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return [cost0, cost1, cost2,cost3], Coef

    def partial_pre(self, X, lr, noise=True,mode=0):  #
        if noise==True:
            X=X+0.15*np.random.randn(*X.shape)
        cost0, _, = self.sess.run((self.reconst_cost_pre, self.optimizer_pre),
                                                                  feed_dict={self.x: X, self.learning_rate: lr})  #
        self.iter = self.iter + 1
        return [cost0]

    def initlization(self):
        tf.reset_default_graph()
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})
  
    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print("model restored")
pred = False
   
def test_face(Img, Label, AE, num_class,lr1=1e-3,lr2=1e-4,pre_iters=200,fit_iters=50,noise=True):
    d = 4           
    alpha = 5
    ro = 0.12
    for i in range(0,1):
        face_10_subjs=Img
        label_10_subjs=Label
        
        AE.initlization()        
        global pred
        if pred==True:
            AE.restore()
        else:
            epoch = 0
            pbatch_size =128
            while epoch < pre_iters:
                indices = np.arange(0, Img.shape[0])
                np.random.shuffle(indices)
                indices = indices[:pbatch_size]
                face_10_subjs_pre = np.array(Img[indices, :])
                cost = AE.partial_pre(face_10_subjs_pre, lr1,noise) 
                epoch = epoch + 1
                if epoch % 50 == 0:#50
                    print("pre epoch: %.1d" % epoch, "cost: %.8f" % (cost[0]/float(batch_size)))
            AE.save_model()   
            pred = True

        display_step = 10
        # fine-tune network
        epoch = 0
        COLD = None
        lastr = 1.0

        while epoch < fit_iters:
            epoch = epoch + 1
            cost, Coef = AE.partial_fit(face_10_subjs, lr2,noise,mode = 'fine')  #
            if epoch % display_step == 0:
                print("epoch: %.1d" % epoch, "cost: %.8f" % (cost[0]/float(batch_size))   )
                #print(cost)
                for posti in range(1):
                    display(Coef, label_10_subjs, d, alpha, ro)
            if COLD is not None:
                normc = np.linalg.norm(COLD, ord='fro')
                normcd = np.linalg.norm(Coef - COLD, ord='fro')#计算Frobenius 范数
                r = normcd/normc
                #print(epoch,r)
                if r < 1.0e-6 and lastr < 1.0e-6:
                    print("early stop")
                    print("epoch: %.1d" % epoch, "cost: %.8f" % (cost[0] / float(batch_size)))
                    #print(cost)
                    for posti in range(1):
                        display(Coef, label_10_subjs, d, alpha, ro)
                    break
                lastr = r
            COLD = Coef
        
        for posti in range(1):
            L,y_pre = display(Coef, label_10_subjs, d, alpha, ro)
            #(Img, Label, AE, 'tsne.png')

    print("%d clusters:" % num_class)    
    visualize(Img, Label, AE)
    return y_pre,L
if __name__ == '__main__':
    data_mat = h5py.File('D:/data/GSM2230760.h5')   
    X=data_mat['X']
    label=data_mat['Y']
    X=np.array(X)
    label=np.array(label)
    num,n_feature=X.shape
    n_clusters = max(label)-min(label)+1#the number of cluster
    
    X=preprocessing(X)
    
    model_path = './models/model-tempfullfc.ckpt'
    restore_path = './models/model-tempfullfc.ckpt'
    logs_path = './logs'

    n_input=[num,n_feature]#num:the number of cell ;n_feature:the number of genes
    n_hidden = [500,500,2000,10]#the size of AE
    
    reg1 = 1.0e-4
    reg2=0.01
    reg3=1
    reg4=1
    lr1=1e-3
    lr2=1e-4#learn rate
    batch_size = num
    
    pre_iters=100#pre_train 
    fit_iters=50
    
    tf.reset_default_graph()
    AE = AE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2,
                 re_constant3=reg3,re_constant4=reg4, batch_size=batch_size,ds=n_clusters,
                 model_path=model_path, restore_path=restore_path, logs_path=logs_path)
    
    y_pred,C= test_face(X, label, AE, n_clusters,lr1,lr2,pre_iters,fit_iters,noise=True)
    NMI=np.round(metrics.normalized_mutual_info_score(label,y_pred),3)
    ARI=np.round(metrics.adjusted_rand_score(label, y_pred),3)
    print('NMI:',NMI,'ARI:',ARI)
