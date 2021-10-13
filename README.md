# Imitating Deep Learning Dynamics via Locally Elastic Stochastic Differential Equations

This repo contains official code for the NeurIPS 2021 paper
**[Imitating Deep Learning Dynamics via Locally Elastic Stochastic Differential Equations](https://arxiv.org/abs/2110.05960v1)** by
[Jiayao Zhang](https://www.jiayao-zhang.com), [Hua Wang](https://statistics.wharton.upenn.edu/profile/wanghua/), [Weijie J. Su](https://statistics.wharton.upenn.edu/profile/suw/).

Discussions welcome, please submit via [Discussions](https://github.com/zjiayao/le_sde/discussions).
You can also read the reviews on [OpenReview](https://openreview.net/forum?id=zEuLFJCRk4X).

```bib
@misc{zhang2021imitating,
      title={Imitating Deep Learning Dynamics via Locally Elastic Stochastic Differential Equations}, 
      author={Jiayao Zhang and Hua Wang and Weijie J. Su},
      year={2021},
      eprint={2110.05960},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Reproducing Experiments
## Dependencies
 
 We use Python 3.8 and ``pytorch`` for training neural nets, please use 
 ``pip install -r requirements.txt`` (potentially in
 a virtual environment) to install dependencies.

## Datasets

  We use a dataset of geometric shapes (GeoMNIST) we constructed as well as CIFAR-10.
  GeoMNIST is lightweighted and will be generated when simulation runs; CIFAR-10 will
  be downloaded from ``torchvision``.

## Code Structure
  
  After instsalling the dependencies, one may navigate through the two
  Jupyter notebooks for running experiments and producing plots and figures.
  Below we outline the code structure.

```
.
├── LICENSE                         # code license
├── README.md                       # this file
├── LE-SDE Data Analysis.ipynb      # reproducing plots and figures
├── LE-SDE Experiments.ipynb        # reproducing experiments
└── src                         # source code
    ├── data_analyzer.py            # processing experiment data
    ├── datasets.py                 # generating and loading datasets
    ├── models.py                   # definition of neural net models
    ├── plotter.py                  # generating plots and figures
    └── utils.py                    # utilities, including training pipelines
└── exp_data                    # experiment data
    ├── *.csv                       # dataframes from neural net training
    └── *.npy                       # numpy.ndarray storing LE-ODE simulations
```

  More info regarding ``npy`` files can be found in the [``numpy`` documentation](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).

# Reproducing Figures

## Experiment Data

  Although all simulations can be run on your machine, it is quite time-consuming.
  Data from our experiments can be downloaded from the following **anonymous** Dropbox links:
  
  - [lesde_exp_data.tar.gz](https://www.dropbox.com/s/kmn08oquefkxqvr/lesde_exp_data.tar.gz?dl=1) (1.02GB): ``*.csv`` files for reproducing Figures 1-4.
  - [lesde_sim_data.tar.gz](https://www.dropbox.com/s/q1ruxc674ye6b3b/lesde_sim_data.tar.gz?dl=1) (2.54GB): ``*.npy`` files for reproducing Figure 5.

  After downloading those tarballs, extract them into ``./exp_data`` (or change the ``EXP_DIR``
  variable in the notebooks accordingly).

  
## Plotter

  Once experiment data are ready, simply follow ``LE-SDE Data Analysis.ipynb`` for
  reproducing all figures.


