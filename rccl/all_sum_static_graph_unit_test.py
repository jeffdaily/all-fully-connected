"""
invocation
HIP_VISIBLE_DEVICES=0,1 python3 all_sum_static_graph_unit_test.py
or
RCCL_TRACE_RT=7 HIP_VISIBLE_DEVICES=0,1 python3 all_sum_static_graph_unit_test.py
"""

import tensorflow as tf
from tensorflow.contrib.rccl import all_sum

with tf.device('/gpu:0'):
    a = tf.get_variable(
        "a", initializer=tf.constant(1.0, shape=(2, 2)))

with tf.device('/gpu:1'):
    b = tf.get_variable(
        "b", initializer=tf.constant(2.0, shape=(2, 2)))

with tf.device('/gpu:0'):
    summed_node = all_sum([a, b])
    
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

with tf.device('/gpu:0'):
    summed = sess.run(summed_node)

print(summed[0])
print(summed[1])

# expected output
# [[3. 3.]
#  [3. 3.]]
# [[3. 3.]
#  [3. 3.]]
