import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import src.data_analyzer as data_analyzer

def viz_imglist(imgs, figsize=None, figname=None, plt=plt):
    """
    **viz_imglist** shows a list of images
    """
    if figsize is not None:
        plt.figure(figsize=figsize)
    for idx, img in enumerate(imgs):
        plt.subplot(1,len(imgs), idx+1)
        ax = plt.gca()
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    if figname is not None:
        plt.tight_layout()
        plt.savefig(figname, bbox_inches='tight')

def get_proj_vec(df, v1=0, v2=1, n_avg=2):
    X_mat = np.mean(np.array(df.sort_values(by=['itr', 'y'])[logit_cols]).reshape(-1, n_classes, n_classes)[-2:], axis=0)
    proj_vec = X_mat[v1] - X_mat[v2]
    return proj_vec / np.linalg.norm(proj_vec)

def viz_sep_1d(exp_data, filters, proj_vec_handle, figsize=(8,6), marker_dict={0: 'o', 1: 's', 2: '^', 3: '*'},
              xlim=(-18,18), filename=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    df = exp_data.hists.query(filters).groupby(['itr', 'y']).mean().reset_index().sort_values(by=['itr', 'y'])
    df_non_group = exp_data.hists.query(filters).sort_values(by=['itr', 'y'])
    T_it = exp_data.T
    n_classes = exp_data.n_classes
    if hasattr(n_classes, "__len__"):
        n_classes = len(df.y.unique())
    plt_itrs = exp_data.plt_itrs
    class_labels = exp_data.class_labels
    logit_cols = [f'y_logit{c}' for c in range(n_classes)]
    X_mat = np.array(df[logit_cols]).reshape(-1, n_classes, n_classes)
    proj_vec = proj_vec_handle(X_mat)

    for idx, it in enumerate(np.linspace(0,T_it-1,50).astype(int)):
        dt = df_non_group[df_non_group.itr == it]
        x_proj = np.array(dt[logit_cols]) @ proj_vec
        ys = np.array(dt.y)
        for c in range(n_classes):
            xx = x_proj[ys == c]
            plt.scatter(xx, [it] * len(xx), marker=marker_dict[c], 
                        color=f'C{c}', alpha=0.2, label=rf'{class_labels[c]}' if idx == 0 else None,
                        s=50)
    
    plt.plot(X_mat[plt_itrs][:,0,:] @ proj_vec, plt_itrs,)
    plt.plot(X_mat[plt_itrs][:,1,:] @ proj_vec, plt_itrs,)
    plt.plot(X_mat[plt_itrs][:,2,:] @ proj_vec, plt_itrs,)

    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel(r'$T_{\nu} \bar{X}^{k}(m)$', size=32)
    plt.ylabel(r'$t$', size=32)
    plt.xlim(xlim)
    plt.legend(fontsize=18, title=rf'').get_title().set_fontsize(24)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

        
def viz_sep_3d(exp_data, filters, figsize=(8,6), marker_dict={0: 'o', 1: 's', 2: '^', 3: '*'},
              view=(15,-10), zinv=False, filename=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    df = exp_data.hists.query(filters).groupby(['itr', 'y']).mean().reset_index().sort_values(by=['itr', 'y'])
    
    T_it = exp_data.T
    n_classes = exp_data.n_classes
    if hasattr(n_classes, "__len__"):
        n_classes = len(df.y.unique())
    plt_itrs = exp_data.plt_itrs
    class_labels = exp_data.class_labels
    logit_cols = [f'y_logit{c}' for c in range(n_classes)]
    X_mat = np.array(df[logit_cols]).reshape(-1, n_classes, n_classes)
    
    plt_itr_max = T_it
    for c in range(n_classes):
        ax.scatter(X_mat[plt_itrs,c,0],X_mat[plt_itrs,c,1],
                   X_mat[plt_itrs,c,2],marker=marker_dict[c],
                   c=np.arange(T_it)[plt_itrs],cmap='viridis_r')

    for idx, it in enumerate(np.linspace(0,T_it-1, 20).astype(int)):
        pnts = X_mat[it]
        ax.plot_trisurf(pnts[:plt_itr_max,0],pnts[:plt_itr_max,1],pnts[:plt_itr_max,2], 
                        alpha=0.1,color=f'C{it}')


    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(T_it, 0))
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, fraction=0.05, shrink=0.7, pad=0.01, )
    ax.tick_params(axis='both', which='minor', bottom=False)


    # ax.view_init(15,-10)
    ax.view_init(*view)
    if zinv:
        ax.invert_zaxis()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

def viz_AB(exp_data, query, est_model, figsize=(8,6), filename=None, ylim=None):
    Ahat, Bhat, _, _ = data_analyzer.get_ABab(exp_data, query, est_model, diff_handle=None)
    plt_itrs = exp_data.plt_itrs
    sns.set_palette('coolwarm',2)
    plt.figure(figsize=figsize)
    plt.scatter(plt_itrs,
                 Ahat[plt_itrs],
                alpha=0.1)
    plt.scatter(plt_itrs,
                Bhat[plt_itrs], 
                alpha=0.1)
    plt.plot(plt_itrs,(data_analyzer.moving_average(Ahat, 10,))[plt_itrs], 
             label=r'$\hat{A}(t)$')
    plt.plot(plt_itrs,(data_analyzer.moving_average(Bhat, 10,))[plt_itrs], 
             label=r'$\hat{B}(t)$')
    plt.legend(fontsize=28, ).get_title().set_fontsize(24)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel(r'$t$', size=32)
    plt.ylabel(r'$\hat{A}(t), \hat{B}(t)$', size=32)
    plt.tight_layout()

    if ylim is not None:
        plt.ylim(ylim)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        
def viz_ab(exp_data, query, est_model, diff_handle, figsize=(8,6), filename=None, ylim=None):
    _, _, dAhat, dBhat = data_analyzer.get_ABab(exp_data, query, est_model, diff_handle=diff_handle)
    plt_itrs = exp_data.plt_itrs
    sns.set_palette('coolwarm',2)
    plt.figure(figsize=figsize)
    
    plt.scatter(plt_itrs,
                 dAhat[plt_itrs],
                alpha=0.05)
    plt.scatter(plt_itrs,
                dBhat[plt_itrs], 
                alpha=0.05)
    plt.plot(plt_itrs, data_analyzer.moving_average(dAhat, 10)[plt_itrs], 
             label=r'$\hat{\alpha}(t)$')
    plt.plot(plt_itrs, data_analyzer.moving_average(dBhat, 10)[plt_itrs],
             label=r'$\hat{\beta}(t)$')
    plt.legend(fontsize=28, ).get_title().set_fontsize(24)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel(r'$t$', size=32)
    plt.ylabel(r'$\hat{\alpha}(t),\hat{\beta}(t)$', size=32)
    plt.tight_layout()

    if ylim is not None:
        plt.ylim(ylim)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

def viz_val_perr(exp_data, metric='valloss', figsize=(8,6), filename=None):
    df = exp_data.hists
    perrs = exp_data.perrs
    plt_itrs = exp_data.plt_itrs
    
    sns.set_palette('coolwarm',n_colors=len(perrs))
    plt.figure(figsize=figsize)
    plt_data = df[(df.itr.isin(plt_itrs)) & (df.p.isin([str(p) for p in perrs]))].rename(
                     columns={'val_loss':'valloss', 'val_acc': 'valacc'}
                 )
    
    ax = plt.gca()
    if metric == 'valloss':
        plt.axhline(y=plt_data[plt_data.itr==0]['valloss'].mean(), c='k', ls=':', lw=3)
    elif metric == 'valacc':
        plt.axhline(y=2/3, c='k', ls=':', lw=3)
    sns.lineplot(x='itr', y=metric, hue='p', data=plt_data)
    
    plt.legend(fontsize=12, title=r'$p_{\mathrm{err}}$').get_title().set_fontsize(24)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel(r'$t$', size=24)
    plt.ylabel('Validation ' + {"valloss": "Loss", "valacc": "Accuracy"}[metric], size=24)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        
def viz_tail_perr(exp_data, est_model='I', metric='valloss', tail_est_kwargs={}, figsize=(8,6), ylim=[0.6, 1.25], filename=None):
    sns.set_palette('coolwarm',2)
    plt.figure(figsize=figsize)
    df = exp_data.hists
    perrs = exp_data.perrs
    plt_itrs = exp_data.plt_itrs

    
    ABs = [data_analyzer.get_ABab(exp_data, query=f'p=="{p}"', model=est_model, diff_handle=None)[:2] for p in perrs]
    tail_A = [data_analyzer.estimate_tail_idx(AB[0], **tail_est_kwargs) for AB in ABs]
    tail_B = [data_analyzer.estimate_tail_idx(AB[1], **tail_est_kwargs) for AB in ABs]
    ax = plt.gca()
    ax.axhspan(0, 1, alpha=0.1, color='green')
    ax.axhspan(1, 2, alpha=0.1, color='red')

    plt.scatter(perrs, tail_A, marker='o', s=100, 
                c='C0', label=r"$\hat{r}_{\alpha}$")
    plt.scatter(perrs, tail_B, marker='x', s=100, 
                c='C1', label=r"$\hat{r}_{\beta}$")
    plt.axhline(y=1, c='k', ls=':', lw=3)
    plt.axvline(x=0.67, c='k', ls=':', lw=3)

    plt.legend(fontsize=28, loc=9).get_title().set_fontsize(24)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel(r'$p_{\mathrm{err}}$ (\% corrupted label)', size=32)
    plt.ylabel(r'Estimated Tail Index', size=32)

    plt.tight_layout()
    if ylim is not None:
        plt.ylim(ylim)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        
def viz_leode(exp_data, filters, sim_data, basename, est_model, n_c=0, align0=0, n_tail=1000, figsize=(8,6), ylim=None, filename=None):
    plt.figure(figsize=figsize)
    
    xbar_sim = sim_data[f'{basename}+{est_model}']
    n_trial, T, _, n_classes = xbar_sim.shape
    plt_itrs = np.hstack([np.arange(0, T, 5), T-1])
    
    df = exp_data.hists.query(filters).groupby(['itr', 'y']).mean().reset_index().sort_values(by=['itr', 'y'])
    logit_cols = [f'y_logit{c}' for c in range(n_classes)]
    Xbar = np.array(df[logit_cols]).reshape(-1,n_classes*n_classes)
    
    sns.set_palette('deep', n_colors=n_classes)
    xs = np.arange(T)
    dat_avg = np.mean(xbar_sim, axis=0) 
    dat_avg *= np.mean(Xbar.reshape(-1,n_classes,n_classes)[-n_tail:,n_c,align0])/np.mean(dat_avg[-n_tail:,n_c,align0]) 
    dat_std = np.std(xbar_sim, axis=0)
    for i in range(n_classes): 
        plt.scatter(plt_itrs, 
                    Xbar.reshape(-1,n_classes,n_classes)[:,n_c,i][plt_itrs], color=f'C{i}',
                 alpha=0.1)
        plt.plot(plt_itrs, dat_avg[:,n_c,i][plt_itrs], c=f'C{i}', 
                 label=r'$\bar{X}^'+rf'{n_c}'+'_'+rf'{i}$',
                 lw=5, alpha=0.8)
        idx = np.linspace(0, len(dat_avg)-1, 20).astype(int)
        plt.errorbar(idx, y=dat_avg[:,n_c,i][idx], 
                     yerr=dat_std[:,n_c,i][idx], alpha=0.8)

    plt.legend(fontsize=28, ).get_title().set_fontsize(24)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel(r'$t$', size=32)
    plt.ylabel(r'$\bar{X}(t)$', size=32)
    plt.tight_layout()

    if ylim is not None:
        plt.ylim(ylim)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
