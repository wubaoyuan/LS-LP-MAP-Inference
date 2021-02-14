# MAP Inference via $ \ell_2 $-Sphere Linear Program Reformulation



## Introduction
This repository provides the implementations using Python and Matlab for our IJCV 2020 work ["MAP Inference via L2-Sphere Linear Program Reformulation"](https://link.springer.com/article/10.1007/s11263-020-01313-2). 

Maximum a posteriori (MAP) inference is an important task for graphical models, which aims to infer the most probable label configuration of a probabilistic graphical mode (e.g, MRF). MAP inference can be reformulated as an integer linear program (ILP) as following: 
$$ \text{MAP}(\boldsymbol{\theta}) = \text{ILP}(\boldsymbol{\theta}) = \mathop{\max}_{\boldsymbol{\mu}, \boldsymbol{v}} < \boldsymbol{\theta}, \boldsymbol{\mu} > ~ \quad \text{s.t.} \quad \boldsymbol{\mu} \in \mathcal{L}_G \cap {0, 1}^{|\boldsymbol{\mu}|}. $$

However, due to integer constraint, the exact optimizer of ILP is intractable in many realistic cases. Therefore, we propose an exact reformulation of the original MAP inference problem. Firstly, we add a new constraint, called $\ell_2$-sphere onto the variable or the factor nodes in order to remove the binary constraint. Then we add an extra variable node to split the $\ell_2$-sphere constraint. 
$$ \text{LS-LP}(\boldsymbol{\theta}) = \mathop{\max}_{\boldsymbol{\mu}, \boldsymbol{v}} < \boldsymbol{\theta}, \boldsymbol{\mu} > ~ \quad \text{s.t.} \quad \boldsymbol{\mu} \in \mathcal{L}_G, \boldsymbol{v} \in \mathcal{S}, \boldsymbol{\mu}_i = \boldsymbol{v}_i, i \in V. $$

It can be proved that $ \text{LS-LP}(\boldsymbol{\theta}) = \text{MAP}(\boldsymbol{\theta}) = \text{ILP}(\boldsymbol{\theta})$, which confirms the feasibility of reformulation. Besides, LS-LP can be efficiently solved by ADMM, which is proved to be globally convergent to epsilon-KKT solution of the original MAP inference.

<div align="center">
<img src="/figure/factor-graph.png" width="500"/>
</div>

## Implementations

#### Python

- Parameter options

``` 
rho_initial: [5e-2, 1e-1, 1e0, 5e0, 1e1, 1e2, 1e3, 1e4]
learning_fact_list: [1.01, 1.03, 1.05, 1.1, 1.2]
rho_upper_bound_list: [1e6, 1e8]
maxIter_list: [500, 1000]
u_factor_solution_list: ['exact-qpc', 'linear-proximal']
dataset_name: ['Grids', 'inpainting4', 'inpainting8', 'scene', 'Segmentation']
file_index: the index of uai file that you want to read and test. Range of index is determined by number of uai files
```

- Demo

```
python run.py \
	--rho_initial 5e-2 \
	--learning_fact 1.01 \
	--rho_upper_bound 1e6 \
	--max_iter 500 \
	--u_factor_solution linear-proximal \
	--dataset_name Grids \
	--file_index 0
```



#### Matlab

- Parameter options

```
Illustration of input variables: rho_initial: {5e-2, 1e-1, 1e0, 5e0, 1e1, 1e2, 1e3, 1e4}
learning_fact_list: {1.01, 1.03, 1.05, 1.1, 1.2}
rho_upper_bound_list: {1e6, 1e8}
maxIter_list: {500, 1000}
u_factor_solution_list: {'exact-qpc', 'linear-proximal'}
dataset_name: {'Grids', 'inpainting4', 'inpainting8', 'scene', 'Segmentation'}
file_index: the index of uai file that you want to read and test. Range of index is determined by number of uai files.
```

- Demo

```
run(5e-2, 1.01, 1e6, 500, 'linear-proximal', 'Grids', 1)
```



## Citation

If our work is helpful to your work, please cite as follows. 

```
@article{wu2020map,
  title={MAP Inference via L2-Sphere Linear Program Reformulation},
  author={Wu, Baoyuan and Shen, Li and Zhang, Tong and Ghanem, Bernard},
  journal={International Journal of Computer Vision},
  pages={1--24},
  year={2020},
  publisher={Springer}
}
```

## [Acknowledgement](#acknowledgement)
[[back to top](#)]

I would like to thank my RAs Wei Sun and Longkang Li for their contributions to the re-organization of Matlab code and the implementation of Python. 





