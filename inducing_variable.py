from gpflow import Parameter
from gpflow.inducing_variables import InducingVariables, InducingPoints
import gpflow 
import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np


class LMCInducingPointsBase(InducingPoints):
    def __init__(self, Z, name = None):
        """
        Inducing points, but the coregionalization indices are not allowed to move (this led to some integer casting bugs at some point).
        :param Z: The initial positions of the inducing points.
        """
        if not isinstance(Z, (tf.Variable, tfp.util.TransformedVariable)):
            if Z.ndim == 2:
                indices = Parameter(Z[:,1][:,None], trainable=False)
                Z = Parameter(Z[:,0][:,None])

        self.Z = tf.concat((Z, indices), 1)

    @property
    def num_inducing(self):
        return tf.shape(self.Z)[0]

    @property
    def shape(self):
        shape = self.Z.shape
        if not shape:
            return None
        return tuple(shape) + (1,)