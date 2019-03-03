import tensorflow as tf
from tensorflow.contrib.nccl import all_sum


def repeat(object, times=None):
    # repeat(10, 3) --> 10 10 10
    if times is None:
        while True:
            yield object
    else:
        for i in range(times):
            yield object


with tf.device('/gpu:0'):
    g0 = tf.placeholder(tf.float32, (2, 2), f"g0")

with tf.device('/gpu:1'):
    g1 = tf.placeholder(tf.float32, (2, 2), f"g1")

all_reduce_sum = all_sum([g0, g1])

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                        allow_soft_placement=False))

init = tf.global_variables_initializer()
sess.run(init)

r = [[1, 1], [1, 1]], [[2, 2], [2, 2]]
for x, y in repeat(r):
    sess.run(all_reduce_sum, feed_dict={g0: x, g1: y})
