"""
invocation
HIP_VISIBLE_DEVICES=0,1 python3 all_sum_static_graph_unit_test.py
or
RCCL_TRACE_RT=7 HIP_VISIBLE_DEVICES=0,1 python3 all_sum_static_graph_unit_test.py
or
HIP_VISIBLE_DEVICES=0,1 python3 all_sum_static_graph_unit_test.py -d <some-large-int-number>
"""

import tensorflow as tf
from tensorflow.contrib.rccl import all_sum
import argparse
import os
import time

parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dim", type=int, default=2,
                    help="one side dimension of the square buffer")
args = parser.parse_args()
print('Dimensions chosen:', args.dim, 'X', args.dim)

with tf.device('/gpu:0'):
    a = tf.get_variable(
        "a", initializer=tf.constant(1.0, shape=(args.dim, args.dim)))

with tf.device('/gpu:1'):
    b = tf.get_variable(
        "b", initializer=tf.constant(2.0, shape=(args.dim, args.dim)))

with tf.device('/gpu:0'):
    summed_node = all_sum([a, b])
    
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True))

init = tf.global_variables_initializer()
sess.run(init)


start = time.clock()
with tf.device('/gpu:0'):
    summed = sess.run(summed_node)
print('Time taken:', time.clock() - start)


print(summed[0])
print(summed[1])


# expected default output
# [[3. 3.]
#  [3. 3.]]
# [[3. 3.]
#  [3. 3.]]
