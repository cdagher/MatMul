from typing import List, Tuple
from jaxtyping import Array

import jax.numpy as jnp
from jax import random as jrandom

import equinox as eqx
from equinox import Module

class split_layer(Module):
    layers: List[Module]

    def __init__(self, layers: List[Module]):
        self.layers = [layer for layer in layers]

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x

    def param_count(self) -> int:
        n = 0
        for layer in self.layers:
            if isinstance(layer, Module):
                if hasattr(layer, 'weight'):
                    shape = list(layer.weight.shape)
                    n += jnp.prod(jnp.array(shape)).item()
                if hasattr(layer, 'use_bias'):
                    if layer.use_bias:
                        n += jnp.sum(jnp.array(list(layer.bias.shape))).item()
        return n
    
    def layers(self) -> List[Module]:
        return self.layers
    
    @classmethod
    def from_layer(cls, W: Module, cut: int):
        layers = []

        U, S, V = jnp.linalg.svd(W.weight)

        W_l_p = jnp.zeros((U.shape[0], cut))
        W_l_pp = jnp.zeros((cut, V.shape[0]))

        for i in range(cut):
            W_l_p = W_l_p.at[:, i].set(U[:, i] * jnp.sqrt(S[i]))
            W_l_pp = W_l_pp.at[i, :].set(V[i, :] * jnp.sqrt(S[i]))

        split_1 = eqx.nn.Linear(W_l_p.shape[1], W_l_p.shape[0], use_bias=W.use_bias, key=jrandom.PRNGKey(0))
        split_2 = eqx.nn.Linear(W_l_pp.shape[1], W_l_pp.shape[0], use_bias=False, key=jrandom.PRNGKey(0))

        split_1 = eqx.tree_at(lambda x: x.weight, split_1, W_l_p)
        if W.use_bias:
            split_1 = eqx.tree_at(lambda x: x.bias, split_1, W.bias)
        split_2 = eqx.tree_at(lambda x: x.weight, split_2, W_l_pp)

        return cls([split_2, split_1])