import tensorflow as tf
from tensorflow.python.client import timeline
import argparse
import os
import time

"""
Copies d-squared float64s (specified with -d option) from gpu:0 to gpu:1
run like this:
python3 gpu-gpu-copy-path.py -d 10000
"""

parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dim", type=int, default=2,
                    help="one side dimension of the square tensor")
args = parser.parse_args()
print('Dimensions chosen:', args.dim, 'X', args.dim)

# var0 is on gpu:0 and initialized to 1s
with tf.device('/gpu:0'):
    var0 = tf.get_variable(
        "var0", initializer=tf.constant(1.0,
                                        shape=(args.dim, args.dim),
                                        dtype=tf.float64))

# var1 on gpu:1 and initialized to random values
with tf.device('/gpu:1'):
    var1 = tf.get_variable(
        "var1", initializer=tf.random.normal(shape=(args.dim, args.dim),
                                             dtype=tf.float64))

# Define op to initilize 
init_op = tf.initialize_all_variables()

# Define graph to copy from gpu0 to gpu1
assign_op = tf.assign(var1, var0)


with tf.Session() as sess:
    # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    sess.run(init_op)

    print("Before, var0")
    print(sess.run(var0))

    print("Before, var1")
    print(sess.run(var1))

    start = time.clock()
    sess.run(assign_op, options=options, run_metadata=run_metadata) #  run with trace
    time_taken = time.clock() - start

    # print("After copy, var1")
    # print(sess.run(assign_op)) # rerun to show correctness
    # print('Time taken:', time_taken)

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_gpu_copy.json', 'w') as f:
        f.write(chrome_trace)
