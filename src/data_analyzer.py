import pandas as pd
import numpy as np
import functools
import scipy.signal

import hashlib
def hash_pd(df):
    return str(hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest())
def get_ABab(exp_data, query, model, diff_handle=None, clean_cache=False):
    if clean_cache or not hasattr(get_ABab, 'cache'):
        get_ABab.cache = {}
    cache = get_ABab.cache
    
    df = exp_data.hists.query(query).groupby([ 'itr', 'y']).mean()
    df_hash = hash_pd(df)
    if df_hash not in cache:
        cache[df_hash] = {}
        
    n_classes = exp_data.n_classes
    if hasattr(n_classes, "__len__"):
        n_classes = len(df.reset_index().y.unique())
        
    if model.upper() in cache[df_hash]:
        Ahat, Bhat = cache[df_hash][model.upper()]
    elif model.upper() == 'L':
        Ahat, Bhat, _, _ = estimate_AB_L(df, n_classes)
        cache[df_hash][model.upper()] = Ahat, Bhat
    else:
        Ahat, Bhat = estimate_AB_I(df, n_classes)
        cache[df_hash][model.upper()] = Ahat, Bhat
    dAhat, dBhat = None, None
    if diff_handle is not None:
        dAhat = diff_handle(Ahat)
        dBhat = diff_handle(Bhat)
    return Ahat, Bhat, dAhat, dBhat

def get_normal(a,b,c):
    n = np.cross(a-b,c-b)
    if np.linalg.norm(n) != 0:
        n /= np.linalg.norm(n)
    return n
def get_xyz(dat, xlim, ylim, pnts, centered=False):
    # dat: 3 * dim
    xx, yy = np.meshgrid(np.linspace(*xlim,pnts),np.linspace(*ylim, pnts))
    normal_vec = get_normal(*dat)
    center_vec = np.mean(dat,axis=0) if not centered else np.zeros(normal_vec.shape)
    zz = (-normal_vec[0] * xx - normal_vec[1] * yy + center_vec @ normal_vec) * 1. /normal_vec[2]
    return xx, yy, zz

def min_max(dat):
    return np.array([np.min(dat), np.max(dat)])

def estimate_AB_L(df, n_classes):
    """
    Xbar: 2darray n_itr * (n_classes * n_features) (n_features = n_classes)
    
    X(t) = c0 + C1 d phi_1 + (C21 + C22) f phi_2
    """
    logits_cols = [f'y_logit{c}' for c in range(n_classes)]
    Xbar = np.array(df[logits_cols]).reshape(-1,9)
    ds = np.array( [[-1/n_classes + (1 if i == j else 0) for j in range(n_classes)] 
                for i in range(n_classes)] ).reshape(-1)
    # estimate along the d direction
    Xbar_s1 = np.sum(Xbar[:,(0,4)], axis=1)-np.sum(Xbar[:,(6,7)], axis=1)
    Xbar_s2 = -np.sum(Xbar[:,(1,2,3,5,6,7)], axis=1)
#     Xbar_s = (Xbar_s1+Xbar_s2)/4
#     Xbar_s_c1 = Xbar_s
#     c1 = Xbar_s[0]
#     Xbar_s /= c1
    Xbar_s = (Xbar_s1+Xbar_s2)/2
    Xbar_s_c1 = Xbar_s / 2
    c1 = Xbar_s[0]
    Xbar_s /= c1
    A_B1 = np.log(np.abs(Xbar_s)) * n_classes
    
    # estimate along the f direection
    Xbar_fc0 = Xbar - np.outer(Xbar_s_c1, ds)
    Xbar_f=Xbar_fc0 @ np.array([2,-1,-1, -1,2,-1, 0,0,0,]) / 3
    Xbar_f /= Xbar_f[0]
    A_B2 = np.log(np.abs(Xbar_f)) * n_classes
    Ahat = (A_B1 + (n_classes-1) * A_B2) / n_classes
    Bhat = (A_B2 - A_B1) *(n_classes-1) / n_classes
    
    return Ahat, Bhat, A_B1, c1

def estimate_AB_I(df, nc):
    logits_cols = [f'y_logit{c}' for c in range(nc)]
    # class_idx * itr * output_dim (K * T * K)
    df = df.reset_index()
    logits = np.array([df[df.y==c][logits_cols] for c in range(nc)])
    
    # avg over class, shape: T * K
    logits_ck = np.mean(logits, axis=0)
    
    c0 = np.nanmean(logits_ck[0:1], axis=0) 
    cks = np.array([np.nanmean(logits[c][0:1], axis=0)-c0 for c in range(nc)])

    B_hat_exp = np.array([(c0 / cks[c]) * (logits[c] - logits_ck) / logits_ck for c in range(nc)])
    B_hat = np.nanmean(np.nanmean(-np.log(np.abs(B_hat_exp)), axis=0), axis=-1)
    A_hat_exp = np.array([(logits_ck * (logits[c] - logits_ck) ** (nc-1)) / (c0 * cks[c] ** (nc-1)) for c in range(nc)])
    A_hat = np.nanmean(np.nanmean(np.log(np.abs(A_hat_exp)), axis=0), axis=-1)
    return A_hat, B_hat

def estimate_tail_idx(dat, tail_n=1000):
    if np.sum(dat[-tail_n:] < 0) > tail_n/2:
        dat = -dat
    return 1-np.nanmean((np.log(dat) / np.log(np.arange(1,len(dat)+1)))[-tail_n:])

nanmean = lambda x : np.nanmean(np.ma.masked_invalid(x))

def diff(arr):
    return (arr[2:] - arr[:-2]) / 2

def moving_average(x, w, **kwargs):
    return np.array(pd.Series(x).rolling(window=w, min_periods=1,
                                         center=True, **kwargs).mean())


def savgol_diff(dat, w=151, order=2):
    return scipy.signal.savgol_filter(dat, window_length=w, polyorder=order, deriv=1)

def estimate_tail_r(dat, tail_n=1000):
    if np.sum(dat[-tail_n:] < 0) > tail_n/2:
        dat = -dat
    return 1-np.nanmean((np.log(dat) / np.log(np.arange(1,len(dat)+1)))[-tail_n:])


def df_to_logits(df, n_c, id_vars=['trial', 'itr']):
    dfs = [df[df.y == c][id_vars +  [f'y_logit{c}']].astype({"trial": int, "itr": int}) for c in range(n_c)]
    df_logits = functools.reduce(lambda df1, df2 : pd.merge(df1, df2, how='outer', on=id_vars), dfs)
    return pd.melt(df_logits, id_vars=id_vars, value_vars=[f'y_logit{c}' for c in range(n_c)]).rename(columns={
        'variable':'Class', 'value':'Logit'
    })

def df_add_smp(df, n, col='n'):
    df[col] = n
    return df.astype({col: str})

def merge_hists(hists, n_c, n_smps, class_labels, col='n'):
    def d_with_col(h, n):
        h[col] = n
        for c, y in enumerate(class_labels):
            h.loc[h['Class']==f'y_logit{c}', 'cls'] = c
            h.loc[h['Class']==f'y_logit{c}', 'Class'] = fr'Class {c} ({y})'
        return h.astype({"cls": int, col: str})
    return pd.concat([d_with_col(h, n)  for (h, n) in zip(hists, n_smps)])
        
def merge_hists_K(hists, n_classes, class_labels):
    def d_with_col(h, n_c):
        for c, y in enumerate(class_labels):
            h.loc[h['Class']==f'y_logit{c}', 'cls'] = c
            h.loc[h['Class']==f'y_logit{c}', 'Class'] = fr'Class {c} ({y})'
        return h.astype({"cls": int, 'K': str})
    return pd.concat([d_with_col(h, n_c)  for (h, n_c) in zip(hists, n_classes)])

class ExperimentResult:
    def __init__(self, n_c, n_smps, class_labels, base_name, hists, hists_m, perrs=None):
        self.n_classes = n_c
        self.n_smps = n_smps
        self.class_labels = class_labels
        self.base_name = base_name
        self.hists = hists
        self.hists_m = hists_m
        self.T = hists.itr.max() + 1
        self.plt_itrs = np.hstack([np.arange(0,self.T,5), self.T-1])
        self.perrs = perrs

def load_geom_N(exp_dir, wrap=True, n_classes = 3,
    class_labels=['Ellipsoid', 'Rectangle', 'Triangle'],
    n_smps=[50, 100, 300,600, 1000, 6000],  base_name='geom',
    file_fmt=r'{base_name}_K3_m{n_smp}_it10k_l0.005.csv'):
    
    local_dict = locals()
    histories = [df_add_smp(pd.read_csv(exp_dir / file_fmt.format(n_smp=n_smp, **local_dict) ),n_smp) for n_smp in n_smps]
    hists_m = merge_hists([df_to_logits(h, n_classes) for h in histories], n_classes, n_smps, class_labels)
    hists = pd.concat(histories)
    if wrap:
        return ExperimentResult(n_classes, n_smps, class_labels, base_name, hists, hists_m)

    return n_classes, n_smps, class_labels, base_name, hists, hists_m
    
def load_cifar10_K(exp_dir, wrap=True, n_smps=2500, n_classes=[2, 3, 4],
                   base_name='cifar10', file_fmt=r'{base_name}_K{n_class}_m{n_smps}_it30k_l0.001.csv'):
    class_labels = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    local_dict = locals()
    histories = [df_add_smp(pd.read_csv(exp_dir / file_fmt.format(n_class=n_c, **local_dict)), n_c, 'K') for n_c in n_classes]
    hists_m = merge_hists_K([df_to_logits(h, n_classes[idx], id_vars=['trial', 'itr', 'K']) for idx, h in enumerate(histories)], 
                            n_classes, class_labels)
    hists = pd.concat(histories)
    if wrap:
        return ExperimentResult(n_classes, n_smps, class_labels, base_name, hists, hists_m)
    return n_classes, n_smps, class_labels, base_name, hists, hists_m

def load_geom_p(exp_dir, n_smps=1000, wrap=True, n_classes=3, base_name='geom', perrs=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.67, 0.7, 0.8],
               file_fmt=r'geom_K3_p{p}.csv'):

    class_labels = ['Ellipsoid', 'Rectangle', 'Triangle']
    local_dict = locals()
    histories = [df_add_smp(pd.read_csv(exp_dir / file_fmt.format(p=p, **local_dict)), p, 'p')
             for p in perrs]
    hists_m = merge_hists([df_to_logits(h, n_classes, id_vars=['trial', 'itr', 'p']) for idx, h in enumerate(histories)], 
                            [n_classes] * len(perrs), perrs, class_labels, col='p')
    hists = pd.concat(histories)
    if wrap:
        return ExperimentResult(n_classes, n_smps, class_labels, base_name, hists, hists_m, perrs)
    return n_classes, n_smps, class_labels, base_name, hists, hists_m, perrs


def load_geom_rd_p(exp_dir, n_smps=1000, wrap=True, n_classes=3, base_name='geom', perrs=[0, 0.1, 0.2, 0.5, 0.8],
               file_fmt=r'geom_K3_rdH_p{p}.csv'):
    class_labels = ['Ellipsoid', 'Rectangle', 'Triangle']
    local_dict = locals()
    histories = [df_add_smp(pd.read_csv(exp_dir / file_fmt.format(p=p, **local_dict)), p, 'p')
             for p in perrs]
    hists_m = merge_hists([df_to_logits(h, n_classes, id_vars=['trial', 'itr', 'p']) for idx, h in enumerate(histories)], 
                            [n_classes] * len(perrs), perrs, class_labels, col='p')
    hists = pd.concat(histories)
    if wrap:
        return ExperimentResult(n_classes, n_smps, class_labels, base_name, hists, hists_m, perrs)
    return n_classes, n_smps, class_labels, base_name, hists, hists_m, perrs


