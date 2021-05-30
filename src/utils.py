import multiprocessing, os, sys
import numpy as np
import torch
import pandas as pd
import tqdm
import src.data_analyzer as data_analyzer
import src.models as models


def console_log(msg, end='\n'):
    os.write(1, ('[LOG/{}]'.format(multiprocessing.current_process().name)+msg+end).encode('utf-8'))

def train(data_set, n_trials, n_iters, save_file, lr, 
          device, n_channels=1,
          verbose=True, last_n_trial=0, bs=None, model_kws={}):
    
    save_times = np.linspace(last_n_trial,last_n_trial+n_trials-1, 20).astype(int)
    xtr, ytr, xval, yval = [torch.Tensor(dt).to(device) for dt in data_set]
    xtr = xtr.double()
    ytr = ytr.long()
    xval = xval.double()
    yval = yval.long()
    n_cls = model_kws['n_classes']
    data_shape = model_kws['data_shape']
    
    if n_channels == 1:
        xtr = xtr.view(-1, 1, *data_shape).double()
        xval = xval.view(-1, 1, *data_shape).double()
    
    N_tr = len(xtr)
    N_val = len(xval)
    
    df_cols = ['trial', 'itr', 'smp', 'y', ] + [f'y_logit{i}' for i in range(n_cls)] + ['tr_loss', 'val_loss', 'val_acc']
    history = pd.DataFrame(columns=df_cols)

    for nt in range(last_n_trial,last_n_trial+n_trials):
        if verbose:
            tqdmr = tqdm.tqdm(range(n_iters))
        else:
            tqdmr = range(n_iters)
        model, loss_func, opt = models.gen_model(**model_kws)
#         model, loss_func, opt = gen_model(n_classes=n_classes, model_func=model_func, lr=lr)
        model.to(device)
        for it in tqdmr:
            if bs is not None:
                ytr_np = data_set[1]
                smp_idx = np.hstack([np.random.choice(np.where(ytr_np == c)[0], bs
                                                     ) for c in range(n_cls)])
                
            else:
                smp_idx = np.random.choice(N_tr, 1)
            xx = xtr[smp_idx]
            yy = ytr[smp_idx]

            model.train()
            ypred = model(xx)
            loss = loss_func(ypred, yy)
            tr_loss = loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()

            model.eval()
            with torch.no_grad():
                model_out = model(xval)
                val_ls = (loss_func(model_out, yval) * xval.shape[0]).item()
                cor = torch.sum(torch.max(model_out,-1)[1] == yval).item()

            val_loss = val_ls / N_val
            val_acc = cor / N_val

            history = history.append(pd.DataFrame([[nt, it, smp_idx[0], yy.item() if bs is None else bs, 
                                                    *[ypred[0][c].item() for c in range(n_cls)],
                                                    tr_loss, val_loss, val_acc]], columns=df_cols))

            if verbose:
                tqdmr.set_description(f"Itr: {it:03d} Smp: {smp_idx[0]:05d} tr_loss: {tr_loss:.2f} val_acc: {val_acc:.2f} val_loss: {val_loss:.3f}")
                tqdmr.refresh()
            else:
                if it % 500 == 0:
                    console_log(f"Itr: {it:03d} Smp: {smp_idx[0]:05d} tr_loss: {tr_loss:.2f} val_acc: {val_acc:.2f} val_loss: {val_loss:.3f}")
        if nt in save_times:
            history.to_csv(save_file, header=True, index=None)
            
    history.to_csv(save_file, header=True, index=None)



def fwd_euler(x0, n_trial, T, f_ahat, f_bhat, noise_scale=1):
    xxs  = np.arange(T)
    ysim = np.zeros((n_trial, len(xxs), 3, 3))
    I3 = np.eye(3)
    # trial, itr, class, logit
    sigmas = [np.linalg.norm(x0[c]) / np.sqrt(3) * noise_scale for c in range(3)]
    for nt in tqdm.tqdm(range(n_trial)):
        ysim[nt][0][0] =  np.random.normal(size=3,scale=sigmas[0])
        ysim[nt][0][1] =  np.random.normal(size=3,scale=sigmas[1])
        ysim[nt][0][2] =  np.random.normal(size=3,scale=sigmas[2])
        for idx in range(1,T):
            cl_comp = [[1,2], [0,2], [0,1]]
            for c in range(3):
                ylast = [np.outer( I3[cc]-1/3, I3[cc]-1/3 ) / (np.linalg.norm(I3[cc]-1/3) ** 2) @ ysim[nt][idx-1][cc] for cc in range(3)]
                ysim[nt][idx][c] = ysim[nt][idx-1][c] + (f_ahat(idx) * ylast[c] + f_bhat(idx) * sum([ylast[cc] for cc in cl_comp[c]]) )/3
    return ysim

def simluate_leode(exp_data, query, est_model, diff_handle, n_trials=500):
    
    df = exp_data.hists.query(query).groupby(['itr', 'y']).mean().reset_index().sort_values(by=['itr', 'y'])
    n_classes = exp_data.n_classes
    if hasattr(n_classes, "__len__"):
        n_classes = len(df.y.unique())
    T_it = exp_data.T
    logit_cols = [f'y_logit{c}' for c in range(n_classes)]
    Xbar = np.array(df[logit_cols]).reshape(-1,n_classes*n_classes)


    if hasattr(n_classes, "__len__"):
        n_classes = len(df.y.unique())

    
    _, _, ahat, bhat = data_analyzer.get_ABab(exp_data, query, est_model, diff_handle=diff_handle)
    
    xbar_sim = fwd_euler(Xbar[0].reshape(3,3), n_trials, len(ahat), lambda i : ahat[i], lambda i : bhat[i], 1)
    return xbar_sim