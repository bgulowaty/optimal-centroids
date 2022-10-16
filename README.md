# optimal-centroids

Source code for paper [Search-based framework for transparent non-overlapping ensemble models](https://ieeexplore.ieee.org/abstract/document/9892360/) by [B.Gulowaty](https://www.researchgate.net/profile/Bogdan-Gulowaty) and [M.Woźniak](https://www.researchgate.net/profile/Michal-Wozniak-6).

## Usage

Please refer to [this notebook](usage_example.ipynb) for an usage example.


```python
from optimalcentroids.optimal_centroids import run

# define variables
models = run(x_train, y_train, n_trees, max_tree_depth, pop_size, n_gen)
``` 

* Method `run` returns list of models. If, during the optimization process, pareto front was created, then the list contains all models based on the pareto front solutions. Otherwise list constains just one model. 
* Retured models are compatible with Sklearn API



### Parallelization

The inner working are based on [pymoo](https://pymoo.org/) optimization library. 
Method `run` accepts `pymoo_elementwise_runner` keyword argument, which refers to [elementwise evaluation function](https://pymoo.org/problems/parallelization.html). Default one is `LoopedElementwiseEvaluation`. 
