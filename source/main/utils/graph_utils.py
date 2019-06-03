import numpy as np
import tensorflow as tf
import logging


def count_trainable_variables():
    params_count = 0
    for v in tf.trainable_variables():
        v_size = np.prod(v.get_shape().as_list())
        logging.info('-- -- Variable %s contributes %s parameters', v, v_size)
        params_count += v_size
    return params_count


def count_variables_from_list_tensors(list_tensors):
    params_count = 0
    for v in list_tensors:
        v_size = np.prod(v.get_shape().as_list())
        params_count += v_size
    return params_count

