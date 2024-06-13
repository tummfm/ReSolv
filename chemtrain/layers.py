# Copyright 2022 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Jax / Haiku implementation of layers to build the DimeNet++ architecture.

The :ref:`dimenet_building_blocks` take components of
:class:`~chemtrain.sparse_graph.SparseDirectionalGraph` as input. Please refer
to this class for input descriptions.
"""
import haiku as hk
from jax import numpy as jnp


class OrthogonalVarianceScalingInit(hk.initializers.Initializer):
    """Initializer scaling variance of uniform orthogonal matrix distribution.

    Generates a weight matrix with variance according to Glorot initialization.
    Based on a random (semi-)orthogonal matrix. Neural networks are expected to
    learn better when features are decorrelated e.g. stated by
    "Reducing overfitting in deep networks by decorrelating representations".

    The approach is adopted from the original DimeNet and the implementation
    is inspired by Haiku's variance scaling initializer.

    Attributes:
        scale: Variance scaling factor
    """
    def __init__(self, scale=2.):
        """Constructs the OrthogonalVarianceScaling Initializer.

        Args:
            scale: Variance scaling factor
        """
        super().__init__()
        self.scale = scale
        self._orth_init = hk.initializers.Orthogonal()

    def __call__(self, shape, dtype=jnp.float32):
        assert len(shape) == 2
        fan_in, fan_out = shape
        # uniformly distributed orthogonal weight matrix
        w_init = self._orth_init(shape, dtype)
        w_init *= jnp.sqrt(self.scale / (max(1., (fan_in + fan_out))
                                         * jnp.var(w_init)))
        return w_init
