
import numpy as np
import tensorflow as tf

def build_pattern(submodel, pattern_size, filter_index, epochs, step_size):
  # Initiate random noise
  input_img_data = np.random.random((1, pattern_size, pattern_size, 1))
  input_img_data = (input_img_data - 0.5) * 127 + 128.
  # input_img_data = 128. * np.ones((1, pattern_size, pattern_size, 1))

  # Cast random noise from np.float64 to tf.float32 Variable
  input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

  # Iterate gradient ascents
  for _ in range(epochs):
      with tf.GradientTape() as tape:
          outputs = submodel(input_img_data, training=False)
          loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
      grads = tape.gradient(loss_value, input_img_data)
      normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
      input_img_data.assign_add(normalized_grads * step_size)

  return input_img_data[0]

def build_pattern_list(submodel, pattern_size, nb_filter, epochs=100, step_size=1.):
  pattern_list = []
  for i in range(nb_filter):
    pattern_list.append(build_pattern(submodel, pattern_size, i, epochs, step_size))
  
  return pattern_list
