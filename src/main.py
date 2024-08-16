from jaxtyping import Float, Array, Int, PyTree

import jax
import jax.numpy as jnp
from jax import random as jrandom

import equinox as eqx
from equinox import Module

import optax as opt

from nn import *

import numpy as np

from matplotlib import pyplot as plt

import streamlit as st

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 2000
PRINT_EVERY = 100
SEED = 0


# Data
def load_data(b: int = 100, key: jrandom.PRNGKey = jrandom.PRNGKey(0)):
    key, subkey = jrandom.split(key)
    X = jrandom.normal(subkey, (b, 2, 10, 10))
    Y = X[:, 0] @ X[:, 1]

    return X, Y


def loss(model: NN, x: Float[Array, "b 2 10 10"], y: Float[Array, "b 10 10"]) -> Float[Array, ""]:
    y_pred = jax.vmap(model)(x[:, 0], x[:, 1])
    return jnp.mean((y - y_pred) ** 2)
    # return cross_entropy(y, y_pred)

def cross_entropy(y: Float[Array, "b 10 10"], y_pred: Float[Array, "b 10 10"]) -> Float[Array, ""]:
    return -jnp.mean(y * jnp.log(y_pred))

def accuracy(y: Float[Array, "b 10 10"], y_pred: Float[Array, "b 10 10"]) -> Float[Array, ""]:
    return jnp.mean(y == y_pred)

def evaluate(model: NN, b: int = 100, key: jrandom.PRNGKey = jrandom.PRNGKey(0)) -> Float[Array, ""]:
    X, Y = load_data(b, key)
    return loss(model, X, Y)

def train(
        model: NN,
        optim: opt.GradientTransformation,
        steps: int,
        print_every: int
) -> NN:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(
        model: NN,
        opt_state: PyTree,
        x: Float[Array, "b 2 10 10"],
        y: Float[Array, "b 10 10"]
    ) -> NN:
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_step = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_step, loss_value
    
    def infinite_trainloader():
        while True:
            X, Y = load_data(BATCH_SIZE)
            yield X, Y

    for s, (x, y) in zip(range(steps), infinite_trainloader()):
        model, opt_state, loss_value = step(model, opt_state, x, y)

        if s % print_every == 0 or s == steps - 1:
            print(f"Step {s}, Loss: {loss_value}")


if __name__ == '__main__':
    print("[INFO] Training CNN")
    print("[INFO] Network Structure:\n")
    print(CNN(), "\n")

    net = CNN()
    # net = MLP()

    net = train(net, opt.adam(LEARNING_RATE), EPOCHS, PRINT_EVERY)
