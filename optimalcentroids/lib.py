import numpy as np
from box import Box
from toolz.curried import pipe
from more_itertools import grouper

def nn_wrapper(nn):
    return Box({
        "predict": lambda x: nn.kneighbors(x, return_distance=False)
    })


def list_with_repeated_elements(input_list, n_repeated):
    return [val for val in input_list for _ in range(n_repeated)]


def find_closeset_val(arr, val):
    return np.argmin(np.abs(np.array(arr) - val))

def individual_to_centroid(indv, n_dim: int):
    return pipe(
        indv,
        lambda x: grouper(x, n_dim),
        list,
        np.array,
        np.nan_to_num
    )