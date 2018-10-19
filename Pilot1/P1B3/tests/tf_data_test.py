# Use of generator in tf.data
# https://stackoverflow.com/questions/47946413/how-to-convert-a-python-data-generator-to-a-tensorflow-tensor

import tensorflow as tf

def gen():
  for _ in range(10):
    yield 1, 10.0, "foo"
    yield 2, 20.0, "bar"
    yield 3, 30.0, "baz"


dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.float32, tf.string))

iterator = dataset.make_one_shot_iterator()

int_tensor, float_tensor, str_tensor = iterator.get_next()
with tf.Session() as sess:
  for _ in range(11):
    print(sess.run(int_tensor), sess.run(float_tensor), sess.run(str_tensor))

