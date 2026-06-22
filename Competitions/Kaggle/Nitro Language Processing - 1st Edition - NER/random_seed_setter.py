seed = 1000 # TODO: replace this with the random seed of your team!

# Note that this should be the general syntax and serves as a guidance,
# errors might appear because of the versions that these packages use

import numpy as np
np.random.seed(seed)
np.random.RandomState(seed)

import random
random.seed(seed)

import torch
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

import tensorflow as tf
tf.random.set_seed(seed)
tf.set_random_seed(seed)

import jax
jax.random.PRNGKey(seed)