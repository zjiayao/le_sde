import numpy as np
import skimage.draw as skd
import scipy.ndimage as simg
import torch

def get_random_smps(x_tr, y_tr, x_va, y_va, n_tr, n_va, n_tot_tr, n_tot_va, n_c):
    tridxs = [np.random.choice(n_tot_tr, n_tr) for _ in range(n_c)]
    vaidxs = [np.random.choice(n_tot_va, n_va) for _ in range(n_c)]
    Xtr = torch.vstack([x_tr[y_tr==k][tridxs[k]] for k in range(n_c)])
    Ytr = np.repeat(np.arange(n_c), n_tr) 
    Xva = torch.vstack([x_va[y_va==k][vaidxs[k]] for k in range(n_c)])
    Yva = np.repeat(np.arange(n_c), n_va) 
    
    return Xtr, Ytr, Xva, Yva

####
# GEOMNIST
####
class Transform:
    rotation = lambda x, incr : simg.rotate(x, incr, reshape=False)
    shift = lambda x, incr    : simg.shift(x, incr, order=1, mode='nearest')
    gauss = lambda x, sigma   : simg.gaussian_filter(x, sigma=sigma)
    
def pair_shuffle(xs, ys):
#     xs, ys = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
    indices = np.arange(0, len(xs))
    np.random.shuffle(indices)
    return xs[indices], ys[indices]

def continue_apply(xs, start, end, func):
    """
    **continue_apply** apply func to xs with linearly interpolation in [start, end]
    """
    N = xs.shape[0]
    incrs = [(end-start) / (1. * N) * i + start for i in range(N)]
    img = [func(xs[i], incrs[i]) for i in range(N)]
    return np.array(img)

def uniform_smp_apply(xs, start, end, func):
    """
    **uniform_smp_apply** apply func to xs with random selection in [start, end]
    """
    N = xs.shape[0]
    incrs = np.random.rand(N)* (end - start) + start
    img = [func(xs[i], incrs[i]) for i in range(N)]
    return np.array(img)

def circle_factory(n, shape, r=10, e=2, dr=1, **kwargs):
    base_img = np.zeros((n, ) + shape)
    for ii, rr in enumerate(np.linspace(r-dr, r+dr, n)):
        idx = skd.ellipse(shape[0]/2, shape[1]/2, rr, rr * e, shape=shape)
        base_img[ii, idx[0], idx[1]] = 1
    return base_img

def square_factory(n, shape, rx=10, ry=15, dr=1, **kwargs):
    base_img = np.zeros((n, ) + shape)
    for ii, rr in enumerate(np.linspace(-dr, dr, n)):
        idx = skd.rectangle((shape[0]/2-rx+rr, shape[1]/2-ry+rr), (shape[0]/2+rx+rr, shape[1]/2+ry+rr), shape=shape)
        base_img[ii, idx[0].astype(int), idx[1].astype(int)] = 1
    return base_img

def triag_factory(n, shape, rx=15, ry=15, dr=1, **kwargs):
    base_img = np.zeros((n, ) + shape)
    cx, cy = shape[0]/2, shape[1] / 2
    for ii, rr in enumerate(np.linspace(-dr, dr, n)):
        idx = skd.polygon([cx+rr, cx+rx+rr, cx-rx+rr], [cy-ry*np.sqrt(3)+rr, cy+ry+rr, cy+ry+rr], shape=shape)
        base_img[ii, idx[0].astype(int), idx[1].astype(int)] = 1
    return base_img

def build_data(n, shape, label, **kwargs):
    label2fact = {
        'circle'   : circle_factory,
        'square'   : square_factory,
        'triangle' : triag_factory,
    }
    if not label.lower() in label2fact.keys():
        raise Exception(f"Label class {label} not found.")
    factory = label2fact[label.lower()]
    dat = factory(n=n, shape=shape, **kwargs)
    dat = continue_apply(dat, 0, 176, Transform.rotation)
    dat = uniform_smp_apply(dat, 0.9, 1.1, Transform.gauss)
    return dat

def gen_geomnist_data(n_tr, n_val, data_shape):
    n_smp_per_cls = n_tr + n_val
    n_classes = 3
    
    circle = build_data(n_smp_per_cls, data_shape, 'circle', r=6)
    square = build_data(n_smp_per_cls, data_shape, 'square', rx=4, ry=8)
    triangle = build_data(n_smp_per_cls, data_shape, 'triangle', rx=10, ry=8)
    
    np.random.shuffle(circle)
    np.random.shuffle(square)
    np.random.shuffle(triangle)
    
    Xtr = np.vstack([triangle[:n_tr], square[:n_tr], circle[:n_tr]])
    Ytr = np.array([0] * n_tr + [1] * n_tr + [2] * n_tr)
    Ytr = np.repeat(np.arange(n_classes), n_tr) 
    Xval = np.vstack([triangle[n_tr:n_tr+n_val], square[n_tr:n_tr+n_val], circle[n_tr:n_tr+n_val]])
    Yval = np.repeat(np.arange(n_classes), n_val) 

    geomnist_data = [Xtr, Ytr, Xval, Yval]
    
    return n_smp_per_cls, n_classes, geomnist_data

####
# CIFAR10
####
def load_cifar10(path):
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return [torchvision.datasets.CIFAR10(root=str(path),
	    train=True, download=True, transform=transform), 
	    torchvision.datasets.CIFAR10(root=str(path),
		train=False, download=True, transform=transform)]

def cifar_unpack(dt, clss=None):
    xs = torch.stack([xx[0] for xx in dt])
    ys = np.array([xx[1] for xx in dt])
    if clss is not None:
        sel = np.zeros(ys.shape)
        for cr in cls:
            sel |= (sel == cr)
        xs = xs[sel]
        ys = ys[sel]
    return xs, ys

def get_cifar10(cifar10_data):
    cifar10_tr = list(cifar10_data[0])
    cifar10_va = list(cifar10_data[1])
    cifar10_xtr, cifar10_ytr = cifar_unpack(cifar10_tr)
    cifar10_xva, cifar10_yva = cifar_unpack(cifar10_va)
    return [cifar10_xtr, cifar10_ytr, cifar10_xva, cifar10_yva]
