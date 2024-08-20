from typing import List, Tuple, Optional
from jaxtyping import Float, Array

from copy import copy, deepcopy

import jax
import jax.numpy as jnp
from jax import random as jrandom

import equinox as eqx
from equinox import Module

import numpy as np

from src.split_layer import split_layer


class NN(Module):
    '''
    A base class which extends the equinox Module class to provide some useful
    functions for this project.
    '''

    layers: List[Module]

    def flatten(self):
        layers = []
        for l in self.layers:
            if isinstance(l, split_layer):
                layers.append(l.layers[0])
                layers.append(l.layers[1])
            else:
                layers.append(l)

        self.layers.clear()
        self.layers.extend(layers)
    
    def replace_layer(self, index: int, layer: Module):
        self.layers[index] = layer
        self.flatten()

    def split_layer(self, index: int, cut: int):
        layer = self.layers[index]
        if isinstance(layer, eqx.nn.Linear):
            split = split_layer.from_layer(layer, cut)
            self.replace_layer(index, split)
        else:
            raise TypeError('Only Linear layers can be split')
        
    def layer_singular_values(self, index: int) -> Array:
        layer = self.layers[index]
        if isinstance(layer, eqx.nn.Linear) is False:
            raise TypeError('Only Linear layers supported')
        
        weights: Array = layer.weight

        _, s, _ = jnp.linalg.svd(weights)

        return s

    def __copy__(self):
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.__dict__.update(self.__dict__)

        return cpy
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.__dict__.update(self.__dict__)

        memo[id(self)] = cpy

        # for i, layer in enumerate(self.layers):
        #     cpy.replace_layer(i, layer)

        for k, v in self.__dict__.items():
            setattr(cpy, k, deepcopy(v, memo))

        return cpy
    
    def param_count(self) -> int:
        n = 0
        for layer in self.layers:
            if isinstance(layer, eqx.Module):
                if hasattr(layer, 'weight'):
                    shape = list(layer.weight.shape)
                    n += jnp.prod(jnp.array(shape)).item()
                if hasattr(layer, 'use_bias'):
                    if layer.use_bias:
                        n += jnp.sum(jnp.array(list(layer.bias.shape))).item()
                if hasattr(layer, 'param_count'):
                    n += layer.param_count()
        return n
    
    def __str__(self):
        return super().__str__() + f' params: {self.param_count()}'


class CNN(NN):
    
    def __init__(self, key: Optional[jrandom.PRNGKey] = jrandom.PRNGKey(0)):
        keys = jrandom.split(key, 4)

        self.layers = [
            # input: 2x10x10
            eqx.nn.Conv2d(2, 4, kernel_size=4, key=keys[0]), # output: 4x7x7
            jax.nn.relu,
            eqx.nn.Conv2d(4, 6, kernel_size=2, key=keys[1]), # output: 6x6x6
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(6*6*6, 250, key=keys[2]), # output: 250
            jax.nn.relu,
            eqx.nn.Linear(250, 100, key=keys[3]), # output: 100
        ]

    def __call__(self, A: Float[Array, "b 10 10"], B: Float[Array, "b 10 10"]) -> Float[Array, "b 10 10"]:
        x = jnp.stack([A, B], axis=0)
        for layer in self.layers:
            x = layer(x)
        x = jnp.reshape(x, (10, 10))
        return x
    
class MLP(NN):
    
    def __init__(self, key: Optional[jrandom.PRNGKey] = jrandom.PRNGKey(0)):
        keys = jrandom.split(key, 2)

        self.layers = [
            eqx.nn.Linear(200, 100, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(100, 50, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Linear(50, 100, key=keys[1]),
        ]

    def __call__(self, A: Float[Array, "b 10 10"], B: Float[Array, "b 10 10"]) -> Float[Array, "b 10 10"]:
        x = jnp.stack([A, B], axis=0)
        x = jnp.reshape(x, (200,))
        for layer in self.layers:
            x = layer(x)
        x = jnp.reshape(x, (10, 10))
        return x

class MatVec(NN):
    '''
    A feed forward NN which approximates a matrix for matrix-vector multiplication.

    Input is a 100-dimensional vector, output is a 10-dimensional vector.
    '''

    def __init__(self, key: Optional[jrandom.PRNGKey] = jrandom.PRNGKey(0)):
        keys = jrandom.split(key, 2)

        self.layers = [
            eqx.nn.Linear(100, 50, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(50, 10, key=keys[1]),
        ]
