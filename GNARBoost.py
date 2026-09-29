import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix
from scipy.sparse import kron, eye, csr_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
from GNAR import *
from tqdm.notebook import tqdm

def difference_matrix(K, order=1):
    """Return the finite difference matrix of given order for K coefficients."""
    D = np.eye(K)
    for _ in range(order):
        D = np.diff(D, n=1, axis=0)
    return D

def generate_b_splines(x,k,n_splines,clamp=False,x_min=None, x_max=None):
    if x_min is None and x_max is None:
        x_min, x_max = np.min(x), np.max(x)
    knots = np.linspace(x_min,x_max,n_splines-k+1)
    if clamp:
        start_knots = np.repeat(x_min, k)
        end_knots = np.repeat(x_max, k)
    else:
        h = knots[1]-knots[0]
        start_knots = np.array([knots[0]-h*i for i in range(k,0,-1)])
        end_knots = np.array([knots[-1]+h*i for i in range(1,k+1)])
    knots = np.concatenate((start_knots,knots,end_knots))

    basis_matrix = np.zeros((len(x),n_splines))

    for i in range(n_splines):
        coeffs = np.zeros(n_splines)
        coeffs[i] = 1
        spline = BSpline(knots, coeffs, k)
        basis_matrix[:,i] = spline(x)

    return basis_matrix

def tensor_product_basis(B1, B2):
    N, K1 = B1.shape
    _, K2 = B2.shape

    # Create empty result matrix
    B_tensor = np.zeros((N, K1 * K2))

    # Fill in each row with the flattened outer product
    for i in range(N):
        outer = np.outer(B1[i], B2[i])  # shape (K1, K2)
        B_tensor[i] = outer.flatten()   # row vector (K1*K2,)
    
    return B_tensor

def tensor_product_penalty(D1, D2, smooth1=1.0, smooth2=1.0):
    K1 = D1.shape[1]
    K2 = D2.shape[1]
    
    I1 = np.eye(K1)
    I2 = np.eye(K2)

    # Penalty for direction 1 (rows)
    P1 = smooth1 * (np.kron(D1, I2).T @ np.kron(D1, I2))
    
    # Penalty for direction 2 (columns)
    P2 = smooth2 * (np.kron(I1, D2).T @ np.kron(I1, D2))
    
    return P1 + P2

class PSplines:
    def __init__(self,x,k,n_splines,smooth,shrink=1.0,clamp=False):
        x = x.reshape(-1)
        self.k = k
        self.n_splines = n_splines
        self.smooth = smooth
        self.shrink = shrink
        self.clamp = clamp
        self.x_min, self.x_max = np.min(x), np.max(x)
        self.V = csr_matrix(generate_b_splines(x,self.k,self.n_splines,self.clamp)) #basis matrix
        self.D = csr_matrix(difference_matrix(self.n_splines,2))
        self.P = self.smooth*self.D.T@self.D

    def fit(self,y):
        V = self.V.toarray()
        P = self.P.toarray()
        self.coefs = np.linalg.solve(V.T@V+P,V.T@y)
        fitted = self.V@self.coefs
        res = y - self.shrink*fitted
        self.sse = np.sum(res**2)

    def predict(self,x):
        V_pred = generate_b_splines(x,self.k,self.n_splines,self.clamp,self.x_min,self.x_max)
        return self.shrink*V_pred@self.coefs

class PSplines2d:
    def __init__(self,psplines1,psplines2,shrink=1.0):
        self.psplines1 = psplines1
        self.psplines2 = psplines2
        self.shrink = shrink
        self.V = csr_matrix(tensor_product_basis(self.psplines1.V.toarray(),self.psplines2.V.toarray()))
        self.P = csr_matrix(tensor_product_penalty(self.psplines1.D.toarray(), self.psplines2.D.toarray(), self.psplines1.smooth, self.psplines2.smooth))

    def fit(self,y):
        V = self.V.toarray()
        P = self.P.toarray()
        self.coefs = np.linalg.solve(V.T@V+P,V.T@y)
        fitted = V@self.coefs
        res = y - self.shrink*fitted
        self.sse = np.sum(res**2)

    def predict(self,x):
        V_pred1 = generate_b_splines(x[:,0],self.psplines1.k,self.psplines1.n_splines,self.psplines1.clamp,self.psplines1.x_min,self.psplines1.x_max)
        V_pred2 = generate_b_splines(x[:,1],self.psplines2.k,self.psplines2.n_splines,self.psplines2.clamp,self.psplines2.x_min,self.psplines2.x_max)
        V_pred = tensor_product_basis(V_pred1,V_pred2)
        return self.shrink*V_pred@self.coefs

class FrozenSpline:
    __slots__ = ("V","P","D","k","n_splines","clamp","x_min","x_max","shrink","coefs","is2d",
                 "ps1","ps2")
    def __init__(self, base):
        # share big stuff
        self.V = base.V
        self.P = base.P
        self.D = getattr(base, "D", None)
        self.shrink = base.shrink
        self.coefs = base.coefs.copy()   # freeze only this

        # metadata for predict
        if isinstance(base, PSplines):
            self.is2d = False
            self.k, self.n_splines, self.clamp = base.k, base.n_splines, base.clamp
            self.x_min, self.x_max = base.x_min, base.x_max
            self.ps1 = self.ps2 = None
        else:  # PSplines2d
            self.is2d = True
            self.ps1, self.ps2 = base.psplines1, base.psplines2  # share refs (small)
            self.k = self.n_splines = self.clamp = None
            self.x_min = self.x_max = None

    def predict(self, x):
        if not self.is2d:
            V_pred = generate_b_splines(x, self.k, self.n_splines, self.clamp, self.x_min, self.x_max)
            return self.shrink * (V_pred @ self.coefs)
        else:
            V1 = generate_b_splines(x[:,0], self.ps1.k, self.ps1.n_splines, self.ps1.clamp, self.ps1.x_min, self.ps1.x_max)
            V2 = generate_b_splines(x[:,1], self.ps2.k, self.ps2.n_splines, self.ps2.clamp, self.ps2.x_min, self.ps2.x_max)
            V  = tensor_product_basis(V1, V2)
            return self.shrink * (V @ self.coefs)

class GNARJump(NetworkModel):
    def __init__(self, alpha_order, beta_order, intercept=True, global_intercept=False, global_alpha=True, global_beta=True, jump=0):
        super().__init__(alpha_order, beta_order,intercept,global_intercept, global_alpha, global_beta)
        self.coef_index = np.cumsum([[self.alpha_orders[i]]+[beta_order[i]] for i in range(self.alpha_order)])
        self.jump = jump
        
    def fit(self, network, vts, use_ls=True):
        self.network = network
        self.vts = vts
        self.T = len(self.vts)
        self.vts_end = vts[-self.alpha_order:,:]
        self.y = vts[self.alpha_order+self.jump:,:].flatten("F")
        self.X = self.transformVTS(vts[:self.T-self.jump])
        if self.global_alpha:
            if use_ls:
                self.coefs,self.res = np.linalg.lstsq(self.X, self.y, rcond=None)[:2]
            else:
                self.coefs = np.linalg.solve(self.X.T@self.X, self.X.T@self.y)
        else:
            if use_ls:
                self.coefs = sparse.linalg.lsqr(self.X,self.y)[0]
            else:
                self.coefs = sparse.linalg.spsolve(self.X.T@self.X, self.X.T@self.y)
        self.vts_fitted = np.reshape((self.X@self.coefs),(-1,network.size),"F")

    def predict(self, vts, steps=1):
        preds = []
        vts = np.vstack( ( vts,np.zeros((1,self.network.size)) ) )
        X = self.transformVTS(vts)

        for h in range(steps):
            pred = np.reshape(X@self.coefs,(-1,self.network.size),"F")
            preds.append(pred)

            #the following multistep prediction uses predicted values instead of real values
            alpha_order_save  = self.alpha_order
            alpha_orders_save = [x for x in self.alpha_orders]
            beta_order_save  = [x for x in self.beta_order]
            intercept_save = self.intercept
            self.alpha_order = alpha_order_save-1
            self.alpha_orders = alpha_orders_save[1:]
            self.beta_order = beta_order_save[1:]
            
            X_old = self.transformVTS(vts[1:])
            if alpha_orders_save[0] == 0 and beta_order_save[0] == 0:
                X = X_old 
            else:
                self.alpha_order = 1
                self.alpha_orders = alpha_orders_save[:1]
                self.beta_order = beta_order_save[:1]
                self.intercept = False
                vts_new = np.vstack( ( pred,np.zeros((1,self.network.size)) ) )
                X_new = self.transformVTS(vts_new)
                X = np.hstack((X_new,X_old))

            self.alpha_order = alpha_order_save
            self.alpha_orders = alpha_orders_save
            self.beta_order = beta_order_save
            self.intercept = intercept_save

        return np.array(preds)

from copy import deepcopy, copy
import gc
from concurrent.futures import ThreadPoolExecutor


class GNARBoost(NetworkModel):
    def __init__(self, alpha_order, beta_order, intercept=True, global_intercept=False, global_alpha=True, global_beta=True,H=1,I=None,recursive_pred=False, 
                 k=3, n_splines_1d=20, n_splines_2d=10, smooth_1d=1., smooth_2d=1., shrink=1.0, uni_term=True, cross_term=True, base_gnar=True, clamp=False, aggregate=False):
        super().__init__(alpha_order, beta_order,intercept,global_intercept, global_alpha, global_beta)
        self.coef_index = np.cumsum([[self.alpha_orders[i]]+[beta_order[i]] for i in range(self.alpha_order)])
        self.coef_order = self.coef_index[-1]
        self.H = H
        if I is None:
            self.I = [100 for h in range(H)]
        else:
            self.I = I
        self.I_eff = [x for x in self.I]
        self.recursive_pred = recursive_pred
        self.k = k
        self.n_splines_1d = n_splines_1d
        self.n_splines_2d = n_splines_2d
        self.smooth_1d=smooth_1d
        self.smooth_2d=smooth_2d
        self.shrink = shrink
        self.uni_term=uni_term
        self.cross_term=cross_term
        self.base_gnar = base_gnar
        self.clamp = clamp
        self.aggregate = aggregate
        if self.recursive_pred:
            self.gnar = GNARJump(self.alpha_order,self.beta_order,self.intercept,self.global_intercept,self.global_alpha,self.global_beta)
        else:
            self.gnars = [GNARJump(self.alpha_order,self.beta_order,self.intercept,self.global_intercept,self.global_alpha,self.global_beta,h) for h in range(self.H)]

    def generate_cross_splines(self,splines):
        splines_2d = [PSplines2d(splines[i],splines[j],self.shrink) 
                                 for i in range(self.coef_order)
                                 for j in range(i+1, self.coef_order)]
        return splines_2d

    def generate_inds(self):
        inds = []
        if self.uni_term:
            inds += [i for i in range(self.coef_order)]
        if self.cross_term:
            inds += [[i,j] for i in range(self.coef_order) for j in range(i+1, self.coef_order)]
        return inds

    def aggregate_splines(self,h):
        inds_opts_new = []
        spline_opts_new = []
        for i in range(self.I[h]):
            inds = self.inds_opts[h][i]
            spline = self.spline_opts[h][i]
            if inds not in inds_opts_new:
                inds_opts_new.append(inds)
                spline_opts_new.append(spline)
            else:
                j = inds_opts_new.index(inds)
                spline_opts_new[j].coefs += spline.coefs
        self.inds_opts[h] = inds_opts_new
        self.spline_opts[h] = spline_opts_new
        gc.collect()
        self.I_eff[h] = len(self.inds_opts[h])
                

    def boost(self, X, res, h, coefs_opts=None, inds_opts=None):
        splines = []
        if self.uni_term:
            splines += [PSplines(X[:,ind],self.k,self.n_splines_1d,self.smooth_1d,self.shrink,self.clamp) for ind in range(self.coef_order)]
        if self.cross_term:
            splines = splines+self.generate_cross_splines(
                [PSplines(X[:,ind],self.k,self.n_splines_2d,self.smooth_2d,self.shrink,self.clamp) for ind in range(self.coef_order)])
        inds = self.generate_inds()
        self.coef_order_boost = len(inds)
        #for i in range(len(splines)):
            #V = splines[i].V.toarray()
            #P = splines[i].P.toarray()
            #print(np.trace(V@np.linalg.solve(V.T@V+P,V.T)),inds[i])

        fitted_len = 0
        if (coefs_opts is not None) and (inds_opts is not None):
            fitted_len = len(coefs_opts[h])
            if self.I[h] < fitted_len:
                self.I[h] = fitted_len
            for i in tqdm(range(fitted_len)):
                ind = inds.index(inds_opts[h][i])
                spline_opt = copy(splines[ind])
                spline_opt.coefs = coefs_opts[h][i]
                res = res - spline_opt.shrink*spline_opt.V.toarray()@spline_opt.coefs
                self.spline_opts[h].append(spline_opt)
            self.inds_opts[h] = inds_opts[h]
        
        for i in tqdm(range(fitted_len, self.I[h]), position=1, leave=False):
            def fit_and_get_sse(spline):
                spline.fit(res)
                return spline.sse
        
            with ThreadPoolExecutor() as executor:
                errors = list(executor.map(fit_and_get_sse, splines))
        
            #spline_opt = deepcopy(splines[np.argmin(errors)])
            spline_opt = FrozenSpline(splines[np.argmin(errors)])
            self.inds_opts[h].append(inds[np.argmin(errors)])
            res -= spline_opt.shrink*spline_opt.V.toarray()@spline_opt.coefs
            self.spline_opts[h].append(spline_opt)
        del splines
        gc.collect()
        self.fit_sse = np.sum((res**2))

    def fit(self, network, vts, use_ls=True, coefs_opts=None, inds_opts=None):
        # the order of cols of vts must match the order of network nodes
        # i.e. the first vts col is the ts for the first node in the network, etc
        self.network = network
        self.vts = vts
        self.T = self.vts.shape[0]-self.alpha_order
        self.N = self.network.size
        self.spline_opts = [[] for h in range(self.H)]
        self.inds_opts = [[] for h in range(self.H)]
        
        if self.recursive_pred:
            self.gnar.fit(self.network,self.vts,use_ls)
            preds = self.gnar.predict(vts[:-1],self.H)
        
        for h in tqdm(range(self.H),position=0,leave=False):
            if self.recursive_pred:
                if self.base_gnar:
                    pred = preds[h][:self.T-h]
                    res = self.gnar.y[h:]- pred.reshape(-1,order="F")
                else:
                    res = self.gnar.y[h:]
                X = self.gnar.transformVTS(vts[:self.T-h])
            else:
                self.gnars[h].fit(self.network,self.vts,use_ls)
                if self.base_gnar:
                    pred = self.gnars[h].predict(vts[:-1-h])[0]
                    res = self.gnars[h].y-pred.reshape(-1,order="F")
                else:
                    res = self.gnars[h].y
                X = self.gnars[h].X
                
            self.boost(X, res, h, coefs_opts, inds_opts)
            
            if self.aggregate:
                self.aggregate_splines(h)

    def CV(self, network, vts, fold_n=5, seed=None, use_ls=True, random=True, full_random=True, base_refit=False, 
           save=False, coefs_opts_folds=None, inds_opts_folds=None, coefs_opts=None, inds_opts=None, last_fit=True):
        self.network = network
        self.vts = vts
        self.T = self.vts.shape[0]-self.alpha_order
        self.N = self.network.size
        self.spline_opts = [[] for h in range(self.H)]
        self.inds_opts = [[] for h in range(self.H)]
        
        if self.recursive_pred:
            self.gnar.fit(self.network,self.vts,use_ls)
            preds = self.gnar.predict(vts[:-1],self.H)

        self.sse_vals_H = [[] for h in range(self.H)] #2d for each h
        
        for h in tqdm(range(self.H)):
            smooth_1d_copy, smooth_2d_copy = self.smooth_1d, self.smooth_2d
            self.smooth_1d, self.smooth_2d = self.smooth_1d*(fold_n-1)/fold_n, self.smooth_2d*(fold_n-1)/fold_n
            if self.recursive_pred:
                pred = preds[h][:self.T-h]
                y = self.gnar.y[h:]
                X = self.gnar.transformVTS(vts[:self.T-h])
            else:
                self.gnars[h].fit(self.network,self.vts,use_ls)
                pred = self.gnars[h].predict(vts[:-1-h])[0]
                y = self.gnars[h].y
                X = self.gnars[h].X
                
            if self.base_gnar:
                res = y-pred.reshape(-1,order="F")
            else:
                res = y

            if random:
                if full_random:
                    fold_inds = random_split_indices(len(res),fold_n,seed)
                else:
                    fold_inds = random_split_indices(int(len(res)/self.N),fold_n,seed)
                    for i in range(len(fold_inds)):
                        fold_inds[i] = np.concatenate([fold_inds[i] + int(n*len(res)/self.N) for n in range(self.N)])
            else:
                fold_inds = np.array_split(np.arange(int(len(res)/self.N)),fold_n)
                for i in range(len(fold_inds)):
                    fold_inds[i] = np.concatenate([fold_inds[i] + int(n*len(res)/self.N) for n in range(self.N)])
            for i in range(len(fold_inds)):
                fold_ind = fold_inds[i]
                X_val = X[fold_ind]
                mask = np.ones(len(res),dtype=bool)
                mask[fold_ind] = False
                X_train = X[mask]
                if self.base_gnar and base_refit:
                    coefs= np.linalg.lstsq(X_train,y[mask], rcond=None)[0]
                    res_train = y[mask] - X_train@coefs
                    res_val = y[fold_ind] - X_val@coefs
                else:
                    res_train = res[mask]
                    res_val = res[fold_ind]
                if coefs_opts_folds is not None and inds_opts_folds is not None:
                    self.boost(X_train, res_train, h,coefs_opts_folds[i],inds_opts_folds[i])
                else:
                    self.boost(X_train, res_train, h)
                if save:
                    with open('gnarb_coefs_fold_'+str(i)+'.pkl', 'wb') as f:
                        pickle.dump([self.spline_opts[0][i].coefs for i in range(len(self.spline_opts[0]))], f)
                    
                    with open('gnarb_inds_fold_'+str(i)+'.pkl', 'wb') as f:
                        pickle.dump(self.inds_opts[0], f)
                self.sse_vals_H[h].append(self.validate(X_val,res_val,h))
                self.spline_opts[h] = []
                self.inds_opts[h] = []
                gc.collect()
            self.sse_vals_H[h] = np.array(self.sse_vals_H[h])
            sses_vals_h_sum = np.sum(self.sse_vals_H[h],axis=0)
            self.I[h] = np.argmin(sses_vals_h_sum)+1
            self.I_eff[h] = self.I[h]

            self.smooth_1d, self.smooth_2d = smooth_1d_copy, smooth_2d_copy
            if last_fit:
                self.boost(X, res, h, coefs_opts=coefs_opts, inds_opts = inds_opts)
            
            if self.aggregate:
                self.aggregate_splines(h)

    def validate(self,X,res,h):
        preds = 0
        sse_vals = []
        for i in range(self.I[h]):
            X_temp = X[:,self.inds_opts[h][i]]
            preds += self.spline_opts[h][i].predict(X_temp)
            sse_vals.append( np.sum((res-preds)**2) )
        return sse_vals

    def evaluate(self,X,h):
        preds = 0
        if self.base_gnar:
            if self.recursive_pred:
                preds += X@self.gnar.coefs
            else:
                preds += X@self.gnars[0].coefs
        for i in range(self.I_eff[h]):
            X_temp = X[:,self.inds_opts[h][i]]
            preds += self.spline_opts[h][i].predict(X_temp)
        return preds

    def get_X_basepreds(self,vts):
        preds = [0 for h in range(self.H)]
        vts_pad = np.vstack( ( vts,np.zeros((1,self.network.size)) ) )
        X = self.transformVTS(vts_pad)
        if self.base_gnar:
            if self.recursive_pred:
                preds = self.gnar.predict(vts,self.H)
            else:
                preds = [self.gnars[h].predict(vts) for h in range(self.H)]
        return X,preds
        
    def predict(self,vts,cutoff=None):
        if cutoff is None:
            cutoff = self.I_eff
        X,preds = self.get_X_basepreds(vts)
        for h in range(self.H):
            for i in range(cutoff[h]):
                X_temp = X[:,self.inds_opts[h][i]]
                preds[h] += np.reshape(self.spline_opts[h][i].predict(X_temp),(-1,self.network.size),"F")
        return preds

    def predict_validate(self,vts):
        if self.aggregate:
            raise ValueError("Aggregate must be false.")
        sses=[[] for h in range(self.H)]
        X,preds = self.get_X_basepreds(vts[:-1])
        for h in range(self.H):
            X = X[:len(X)-h]
            if self.base_gnar:
                preds[h] = preds[h][:len(preds[h])-h]
            for i in range(self.I[h]):
                X_temp = X[:,self.inds_opts[h][i]]
                preds[h] += np.reshape(self.spline_opts[h][i].predict(X_temp),(-1,self.network.size),"F")
                sses[h].append(np.sum((vts[self.alpha_order+h:] - preds[h])**2))
        return sses