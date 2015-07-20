# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Tanh transform functions and classes.
"""

from neon.transforms.activation import Activation


class TanhOpt(Activation):

    """
    Embodiment of a optimized tanh activation function.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.gain = 1.0

    def apply_function(self, backend, inputs, outputs):
        """
        Applies the hyperbolic tangent transform to the dataset passed.

        Arguments:
            inputs (array_like): Input data to be transformed

        Returns:
            array_like: Transformed copy of the inputs.  Will be in the same
                        format as the input inputs.
        """
        backend.multiply(inputs, 2.0/3.0, outputs)
        backend.tanh(outputs, outputs)
        backend.multiply(outputs, 1.7159, outputs)

    def apply_derivative(self, backend, inputs, outputs):
        """
        Applies derivative of the hyperbolic tangent transform to the inputs
        passed.

        Arguments:
            inputs (array_like): Input data to be transformed

        Returns:
            array_like: Transformed copy of the inputs.  Will be in the same
                        format as the input inputs.
        """
        self.apply_function(backend, inputs, outputs)
        backend.multiply(outputs, outputs, outputs)
        backend.divide(outputs, 1.7159**2, outputs)
        backend.subtract(1.0, outputs, outputs)
        backend.multiply(outputs, 1.7159*2.0/3.0, outputs)

    def fprop_func(self, backend, inputs, outputs):
        self.apply_function(backend, inputs, outputs)
        backend.multiply(outputs, outputs, inputs)
        backend.divide(inputs, 1.7159**2, inputs)
        backend.subtract(1.0, inputs, inputs)
        backend.multiply(inputs, 1.7159*2.0/3.0, inputs)
