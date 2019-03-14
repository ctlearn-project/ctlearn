import tensorflow as tf
constant = tf.constant([1, 2, 3])
tensor = constant * constant
with tf.Session() as sess:
	print(tensor.eval())
print("akash the great")