from typing import Protocol, Union
import tensorflow as tf
import numpy as np    

class Loss(Protocol):
    def __call__(self, y_true: Union[np.ndarray, tf.Tensor] , y_pred: Union[np.ndarray, tf.Tensor]) -> Union[tf.Tensor, np.ndarray]:
        ...
        