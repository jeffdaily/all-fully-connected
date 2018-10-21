#! /usr/bin/env python

"""Multilayer Perceptron for drug response problem converted to TensorFlow"""

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import p1b3
from pudb import set_trace

# Model and Training parameters
SEED = 2016
BATCH_SIZE = 100
EPOCHS = 20
WORKERS = 1
OUT_DIR = '.'

# Type of feature scaling (options: 'maxabs': to [-1,1]
#                                   'minmax': to [0,1]
#                                   None    : standard normalization
SCALING = 'std'
# Features to (randomly) sample from cell lines or drug descriptors
# FEATURE_SUBSAMPLE = 500
FEATURE_SUBSAMPLE = 0

MIN_LOGCONC = -5.
MAX_LOGCONC = -4.
CATEGORY_CUTOFFS = [0.]
VAL_SPLIT = 0.2
TEST_CELL_SPLIT = 0.15

D1, D2, D3, D4 = 6000, 500, 100, 50

np.set_printoptions(threshold=np.nan)
np.random.seed(SEED)


# TF dataset
def input_fn(data_getter):
    dataset = (tf.data.Dataset.from_generator(
        generator=lambda: data_getter,
        output_types=(tf.float32, tf.float32),
    )
        .repeat()
        .make_one_shot_iterator().get_next()
    )
    return dataset[0], dataset[1]


def fc_model_fn(features, labels, mode):
    """Model function for a fully-connected network"""

    input_layer = tf.reshape(features, [-1, 29532])
    dense_1 = tf.layers.dense(inputs=input_layer, units=D1,
                              activation=tf.nn.relu)
    dense_2 = tf.layers.dense(inputs=dense_1, units=D2,
                              activation=tf.nn.relu)
    dense_3 = tf.layers.dense(inputs=dense_2, units=D3,
                              activation=tf.nn.relu)
    dense_4 = tf.layers.dense(inputs=dense_3, units=D4,
                              activation=tf.nn.relu)

    regressed_val = tf.layers.dense(inputs=dense_4, units=1)

    param_count = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('Total Param Count: {}'.format(param_count))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "output": regressed_val,
    }

    tf.logging.info('mode: {}'.format(mode))
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions['output'])

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, regressed_val)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions['output'])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():

    tf.logging.set_verbosity(tf.logging.DEBUG)

    loader = p1b3.DataLoader(val_split=VAL_SPLIT,
                             test_cell_split=TEST_CELL_SPLIT,
                             cell_features=['expression'],
                             drug_features=['descriptors'],
                             feature_subsample=FEATURE_SUBSAMPLE,
                             scaling=SCALING,
                             scramble=False,
                             min_logconc=MIN_LOGCONC,
                             max_logconc=MAX_LOGCONC,
                             subsample='naive_balancing',
                             category_cutoffs=CATEGORY_CUTOFFS)

    print('Loader input dim', loader.input_dim)

    gen_shape = None

    train_gen = p1b3.DataGenerator(loader, batch_size=BATCH_SIZE,
                                   shape=gen_shape, name='train_gen').flow()
    val_gen = p1b3.DataGenerator(loader, partition='val',
                                 batch_size=BATCH_SIZE,
                                 shape=gen_shape, name='val_gen').flow()
    val_gen2 = p1b3.DataGenerator(loader, partition='val',
                                  batch_size=BATCH_SIZE,
                                  shape=gen_shape, name='val_gen2').flow()
    test_gen = p1b3.DataGenerator(loader, partition='test',
                                  batch_size=BATCH_SIZE,
                                  shape=gen_shape, name='test_gen').flow()

    # Create the Estimator
    p1b3_regressor = tf.estimator.Estimator(
        model_fn=fc_model_fn, model_dir="/tmp/fc_regression_model")


    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(train_gen))
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(val_gen))
    tf.estimator.train_and_evaluate(p1b3_regressor, train_spec, eval_spec)


if __name__ == '__main__':
    main()
