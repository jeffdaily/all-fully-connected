import tensorflow as tf
from tensorflow.contrib.nccl import all_sum

with tf.device('/gpu:0'):
    a = tf.get_variable(
        f"a", initializer=tf.constant(1.0, shape=(2, 2)))

with tf.device('/gpu:1'):
    b = tf.get_variable(
        f"b", initializer=tf.constant(2.0, shape=(2, 2)))

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

with tf.device('/gpu:0'):
    summed = sess.run(all_sum([a, b]))

print(summed[0])
print(summed[1])

# expected output
# [[3. 3.]
#  [3. 3.]]
# [[3. 3.]
#  [3. 3.]]
