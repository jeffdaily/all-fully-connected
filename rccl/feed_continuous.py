import tensorflow as tf


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


with tf.device('/gpu:0'):
    sum = tf.add(g0, g1, 'add_go_g1')

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

# print(sess.run(sum, feed_dict={g0: [[1, 1], [1, 1]], g1: [[2, 2], [3, 3]]}))

r = [[1, 1], [1, 1]], [[2, 2], [2, 2]]
for x, y in repeat(r, 10):
    print(sess.run(sum, feed_dict={g0: x, g1: y}))
