#Copyright (C) 2020 Kasra Arnavaz
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
# Author: Kasra Arnavaz <kasra at di dot ku dot dk>
# 
# 2020-10-19 Kasra Arnavaz <kasra at di dot ku dot dk>

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.optimize import *
from scipy.special import expit
import os


class AL:

    def __init__(self, measure, visualize):
        
        self.measure = measure
        self.visualize = visualize
        np.random.seed(self.init_seed)
        self.seed = np.random.permutation(AL.num_rep)
        self.results = np.zeros([4, self.max_num])
        
    ###########################################################
        
    def toy_data(self):

        def AB():

            x_1 = np.random.multivariate_normal([1,1], 0.5*np.eye(2), size=int(self.num_samples/2))
            t_1 = np.ones(int(self.num_samples/2))
            x_0 = np.random.multivariate_normal([-1,-1], 0.5*np.eye(2), size=int(self.num_samples/2))
            t_0 = np.zeros(int(self.num_samples/2))
            x = np.append(x_1, x_0, axis=0)
            t = np.append(t_1, t_0)
            return x, t
        
        def ABA():
   
            x_a_left = np.random.multivariate_normal([-2, 0],0.5*np.eye(2), size=int(self.num_samples/4))
            x_a_right = np.random.multivariate_normal([2, 0], 0.5*np.eye(2), size=int(self.num_samples/4))
            x_a = np.append(x_a_left, x_a_right, 0)
            np.random.shuffle(x_a)
            t_a = np.ones(int(self.num_samples/2))
            x_b = np.random.multivariate_normal([0,0], 0.5*np.eye(2), size=int(self.num_samples/2))
            t_b = np.zeros(int(self.num_samples/2))
            x = np.append(x_a, x_b, 0)
            t = np.append(t_a, t_b)
            return x, t

        def digits():
            x, t = load_digits(n_class=2, return_X_y=True)
            return x, t
        
        def cancer():
            x, t = load_breast_cancer(True)
            return x, t

        def haberman():
            data = np.loadtxt('data/haberman.data', delimiter=',')
            x, t = data[:, :-1], data[:,-1]
            return x, t

        def heart():
            data = np.loadtxt('data/heart.dat', delimiter=' ')
            x, t = data[:, :-1], data[:,-1]
            return x, t

        def ion():
            data = np.loadtxt('data/ionosphere.data', delimiter=',')
            x, t = data[:, :-1], data[:,-1]
            return x, t

        def parkinsons():
            data = np.loadtxt('data/parkinsons.data', delimiter=',')
            x, t = np.delete(data, -7, 1), data[:,-7]
            return x, t

        def DD():
            x = np.loadtxt('data/DD.txt', delimiter='\t')
            t = np.loadtxt('data/DD_labels.txt', delimiter='\n')
            return x, t

        def aids():
            x = np.loadtxt('data/AIDS_nodedeg_hist_features.data', delimiter=' ')
            t = np.loadtxt('data/AIDS_graph_labels.txt')
            return x, t

        list_names = ['AB', 'ABA', 'digits', 'cancer', 'haberman', 'heart', 'ion', 'parkinsons', 'DD', 'aids']
        list_funcs = [AB, ABA, digits, cancer, haberman, heart, ion, parkinsons, DD, aids]
        for name, func in zip(list_names, list_funcs):
            if name == self.data: return func()

    def principal_components(self, x):
        if x.shape[1] <= 2:
            return x, [0, 0]
        pca = PCA(n_components=2)
        x_trans = pca.fit_transform(x)
        explained_var = pca.explained_variance_ratio_    
        return x_trans, explained_var

    def split(self, x, t, s):
        x_tr, x_ts, t_tr, t_ts = train_test_split(x, t, test_size=0.2, random_state=s)
        scaler = preprocessing.StandardScaler().fit(x_tr)
        x_tr = scaler.transform(x_tr)
        x_ts = scaler.transform(x_ts)
        x_tr = np.append(x_tr, np.ones([x_tr.shape[0], 1]), axis=1)
        x_ts = np.append(x_ts, np.ones([x_ts.shape[0], 1]), axis=1)
        return x_tr, t_tr, x_ts, t_ts



    ###########################################################
            
    def train_model(self, x_tr, t_tr, mask, x_ts, t_ts):
            
        def loss_train(w):
            c0 = np.unique(t_tr)[0]
            mask_c0 = (t_tr == c0)
            t = t_tr.copy()
            t[mask_c0] = -1
            t[~mask_c0] = 1
            a = w.reshape(1,-1)@x_tr[mask].T
            a = a.reshape(-1)
            G = np.log(1/(1+np.exp(-a*t[mask])))
            M = -np.sum(G) + 0.5*self.alpha*np.sum(w**2)
            return M            
        k = np.shape(x_tr)[1]
        res = minimize(loss_train, x0=np.zeros(k), tol=1e-5)
        w_mp = res.x
        def y(x, w):
            a = w.reshape(1,-1)@x.T
            a = a.reshape(-1)
            y = 1/(1+np.exp(-a))
            return y
        m0 = t_ts == np.unique(t_ts)[0]
        t_ts[m0] = 0
        t_ts[~m0] = 1
        accu_ts = np.mean(np.round(expit(x_ts@ w_mp)) == t_ts)
        y_tr = expit(x_tr@ w_mp)
        B_N = np.zeros([k,k])
        for xn, yn in zip(x_tr[mask], y_tr[mask]):
            B_N =  B_N + xn.reshape(-1,1)@xn.reshape(1,-1)*yn*(1-yn)
        A_N = B_N + self.alpha*np.eye(k)
        
############################
        while True:            
            def alpha_eq(alpha):
                return alpha*np.sum(w_mp**2) + alpha*np.trace(np.linalg.inv(B_N + alpha*np.eye(k))) - k
            old_alpha = self.alpha
            self.alpha = fsolve(alpha_eq, .01)
            new_alpha = self.alpha
            gamma = k-self.alpha*np.trace(np.linalg.inv(A_N))
            res = minimize(loss_train, x0=np.zeros(k), tol=1e-5)
            w_mp = res.x
            y_tr = expit(x_tr@ w_mp)
            B_N = np.zeros([k,k])
            for xn, yn in zip(x_tr[mask], y_tr[mask]):
                B_N =  B_N + xn.reshape(-1,1)@xn.reshape(1,-1)*yn*(1-yn)
            if np.abs(old_alpha - new_alpha < 1e-4): break
    
        y_tr = expit(x_tr@ w_mp)
        A_N = np.zeros([k,k])
        for xn, yn in zip(x_tr[mask], y_tr[mask]):
            A_N += xn.reshape(-1,1)@xn.reshape(1,-1)*yn*(1-yn)
        A_N += self.alpha*np.eye(k)
        gamma = k-self.alpha*np.trace(np.linalg.inv(A_N))

######################            
        loss_tr = loss_train(w_mp)
        N = len(mask)
        a_ts = np.sum(w_mp * x_ts, axis=1)
        sq = np.zeros(x_ts.shape[0])
        for n in range(x_ts.shape[0]):
            sq[n] = x_ts[n].reshape(1,-1) @ np.linalg.inv(A_N) @ x_ts[n].reshape(-1,1) 
        ks = 1/(np.sqrt(8+np.pi*sq))
        a_mod = ks * a_ts
        y_mod = expit(a_mod)
        y_ts = expit(x_ts@ w_mp)
##        loss_ts = log_loss(t_ts, y_ts)            
        loss_ts = log_loss(t_ts, y_mod)            
        
        return y_tr, loss_tr, w_mp, A_N, accu_ts, loss_ts

    ###########################################################
          
    def sampling(self, x_tr, t_tr, y_tr, mask, A_N, w_mp, explained_var):

        def init_sampling():
            t0, t1 = np.unique(t_tr)
            mask_0 = np.where(t_tr == t0)[0]
            mask_1 = np.where(t_tr == t1)[0]
            rand_0 = np.random.choice(mask_0)
            rand_1 = np.random.choice(mask_1)
            init = [rand_0, rand_1]
            return init
        
        def random():
            all_idx = np.arange(x_tr.shape[0])
            next_sample = np.random.choice(all_idx)
            mask.append(next_sample)
            return mask
            
        def decision_boundary():
            all_idx = np.arange(x_tr.shape[0])
            measure = y_tr*(1-y_tr)
            next_sample = np.argsort(measure)[-1]
            mask.append(next_sample)
            return mask

        def info_gain():
            N = len(mask)
            k = x_tr.shape[1]
            all_idx = np.arange(x_tr.shape[0])
            norm = np.zeros(x_tr.shape[0])
            for n in range(x_tr.shape[0]):
                norm[n] = x_tr[n].reshape(1,-1) @ np.linalg.inv(A_N) @ x_tr[n].reshape(-1,1) 
            measure = 0.5*np.log(1+y_tr*(1-y_tr)*norm)
            next_sample = np.argsort(measure)[-1]      
            mask.append(next_sample)
            return mask            
        
        if mask is None: return init_sampling()
        else:
            list_names = ['random', 'decision_boundary', 'info_gain']
            list_funcs = [random, decision_boundary, info_gain]
            for name, func in zip(list_names, list_funcs):
                if name == self.measure: return func()

    def data_space(self, x, t, var):
        slack = 1
        x0 = np.linspace(np.amin(x[:,0])-slack, np.amax(x[:,0])+slack, 100)
        x1 = np.linspace(np.amin(x[:,1])-slack, np.amax(x[:,1])+slack, 100)
        X0, X1 = np.meshgrid(x0, x1)
        plt.figure()
        mask_zero = (t == np.unique(t)[0])
        plt.scatter(x[mask_zero,0], x[mask_zero,1], marker = 'o')
        plt.scatter(x[~mask_zero,0], x[~mask_zero,1], marker = 'x')
        plt.xlabel('PC1 (%.2f)'%var[0])
        plt.ylabel('PC2 (%.2f)'%var[1])
        if not os.path.exists('plots/curves/%s'%self.data): os.makedirs('plots/curves/%s'%self.data)
        plt.savefig('plots/curves/%s/%s.pdf'%(self.data, self.data))
        plt.close()

    ###########################################################
    
    def visualize_contours(self, x, t, explained_var, w_mp, A, mask, i):
        slack = 1
        x0 = np.linspace(np.amin(x[:,0])-slack, np.amax(x[:,0])+slack, 100)
        x1 = np.linspace(np.amin(x[:,1])-slack, np.amax(x[:,1])+slack, 100)
        X0, X1 = np.meshgrid(x0, x1)
        
        def decision_boundary(X0, X1):
            a = w_mp[0]*X0 + w_mp[1]*X1 + w_mp[2]
            y = expit(a)
            return y*(1-y)             
        
        def info_gain(X0, X1):
            a = w_mp[0]*X0 + w_mp[1]*X1 + w_mp[2]
            X = np.stack((X0,X1, np.ones_like(X0)), axis=0)
            norm = np.zeros_like(X0)
            for i in range(X0.shape[0]):
                for j in range(X1.shape[1]):
                    norm[i,j] = X[:,i,j].reshape(1,-1)@np.linalg.inv(A)@X[:,i,j].reshape(-1,1)
            y = expit(a)
            mackay = y*(1-y)*(norm)
            delta_s = 0.5*np.log(1+mackay)
            return delta_s
    

        list_names = ['decision_boundary', 'info_gain',]
        list_funcs = [decision_boundary, info_gain,]
        for name, func in zip(list_names, list_funcs):
            if name == self.measure: delta_s = func(X0, X1)
            
        plt.figure()
        mask_zero = (t == np.unique(t)[0])
        plt.contourf(X0, X1, delta_s, 50, cmap='cool', alpha=0.7)
        plt.colorbar()
        plt.scatter(x[mask_zero,0], x[mask_zero,1], c='black', marker = 'o')
        plt.scatter(x[~mask_zero,0], x[~mask_zero,1], c='black', marker = 'x')
        for m in mask:
            if t[m] == np.unique(t)[0]: mark = 'o'
            else: mark = 'x'
            plt.scatter(x[m,0], x[m,1], c='white', marker = mark)
        
        x_lim = plt.xlim()
        y_lim = plt.ylim()
        y_axis = np.array(y_lim)
        plt.plot((-w_mp[2]-w_mp[1]*y_axis)/w_mp[0], y_axis, 'black')
        plt.xlabel('PC1 (%.2f)' %explained_var[0])
        plt.ylabel('PC2 (%.2f)' %explained_var[1])
        plt.title(self.data)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        if not os.path.exists('plots/contours/%s/%s'%(self.data,self.measure)): os.makedirs('plots/contours/%s/%s'%(self.data,self.measure))
        plt.savefig('plots/contours/%s/%s/%s_%s_%d.pdf'%(self.data, self.measure ,self.data, self.measure, i))
        plt.close()

    ###########################################################

    def active_learning(self, x_tr, t_tr, y_tr, mask, A_N, w_mp, explained_var, x_ts, t_ts, s):
        accu_ts, loss_ts = [], []
        np.random.seed(s)
        for i in range(self.max_num):
            self.alpha = np.exp(np.random.uniform(10^-3,10))
            mask = self.sampling(x_tr, t_tr, y_tr, mask, A_N, w_mp, explained_var)
            y_tr, loss_tr, w_mp, A_N, accu_test, loss_test = self.train_model(x_tr, t_tr, mask, x_ts, t_ts)
            accu_ts.append(accu_test)
            loss_ts.append(loss_test)
            if self.visualize:
               self.visualize_contours(x_tr, t_tr, explained_var, w_mp, A_N, mask, i)
            
        return np.array(accu_ts), np.array(loss_ts)


    def repeat(self):
        accu_matrix = np.zeros([self.num_rep, self.max_num])
        loss_matrix = np.zeros([self.num_rep, self.max_num])
        np.random.seed(self.init_seed)
        x, t = self.toy_data()   
        x, explained_var = self.principal_components(x)
        x_tr, t_tr, x_ts, t_ts = self.split(x, t, self.init_seed)
        mask, y_tr, A_N, w_mp = [None]*4
        if self.measure=='random':
            self.data_space(x_tr, t_tr, explained_var)
        print('Running %s sampling on %s...'%(self.measure, self.data))
        for i, s in enumerate(self.seed):
            print('%d/%d'%(i+1, len(self.seed)))
            accu, loss = self.active_learning(x_tr, t_tr, y_tr, mask, A_N, w_mp, explained_var, x_ts, t_ts, s)
            accu_matrix[i] = accu
            loss_matrix[i] = loss     
        self.results[0] = np.mean(accu_matrix, 0)
        self.results[1] = np.std(accu_matrix, 0)
        self.results[2] = np.mean(loss_matrix, 0)
        self.results[3] = np.std(loss_matrix, 0)
        return self
    
     ###########################################################       

    @classmethod
    def learning_curve_error_bar(cls, rand_results, unc_results, tot_results):

        every_20 = np.zeros_like(rand_results[0])
        every_21 = np.zeros_like(rand_results[0])
        every_22 = np.zeros_like(rand_results[0])
        every_20[[0,20,40,60,80]] = 1
        every_21[[1,21,41,61,81]] = 1
        every_22[[2,22,42,62,82]] = 1
   
        plt.figure()
        plt.errorbar(np.arange(2, cls.max_num + 2), rand_results[0],rand_results[1]*every_20, label='random')
        plt.errorbar(np.arange(2, cls.max_num + 2), unc_results[0],unc_results[1]*every_21, label='decision boundary')
        plt.errorbar(np.arange(2, cls.max_num + 2), tot_results[0],tot_results[1]*every_22, label='information gain')
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.xticks([2,20,40,60,80,100])
        plt.title(cls.data)
        plt.savefig('plots/curves/%s/rep_accu_%s_%d_%d.pdf'%(cls.data, cls.data, cls.max_num, cls.num_rep))
        plt.close()

        plt.figure()
        plt.errorbar(np.arange(2, cls.max_num + 2), rand_results[2],rand_results[3]*every_20, label='random')
        plt.errorbar(np.arange(2, cls.max_num + 2), unc_results[2],unc_results[3]*every_21, label='decision boundary')
        plt.errorbar(np.arange(2, cls.max_num + 2), tot_results[2],tot_results[3]*every_22, label='information gain')
        plt.xlabel('Number of  training samples')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.xticks([2,20,40,60,80,100])
        plt.title(cls.data)
        plt.savefig('plots/curves/%s/rep_loss_%s_%d_%d.pdf'%(cls.data, cls.data, cls.max_num, cls.num_rep))
        plt.close()

    ###########################################################
        
for data in ['AB', 'ABA', 'digits', 'cancer', 'haberman', 'heart', 'ion', 'parkinsons', 'DD', 'aids']:
    AL.data = data
    AL.max_num = 99
    AL.num_samples = 200
    AL.num_rep = 20
    AL.init_seed = np.random.randint(AL.num_rep)
    AL.init_seed = 619
    rand = AL('random', False)
    db = AL('decision_boundary', True)
    ig = AL('info_gain', True)
    rand.repeat()
    db.repeat()
    ig.repeat()
    AL.learning_curve_error_bar(rand.results, db.results, ig.results)
