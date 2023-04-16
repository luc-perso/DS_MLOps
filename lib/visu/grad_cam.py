import numpy as np
import tensorflow as tf
import cv2


def grad_cam(img, grad_model, resize_shape=None, pred_index=None):
  image_size = img.shape[0]

  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(np.expand_dims(img, axis=0), training=False)
    if pred_index is None:
      pred_index = tf.argmax(predictions[0])
    loss = predictions[:, pred_index]
  
  output = conv_outputs[0]
  grads = tape.gradient(loss, conv_outputs)[0]

  gate_f = tf.cast(output > 0, 'float32')
  gate_r = tf.cast(grads > 0, 'float32')
  guided_grads = gate_f * gate_r * grads

  weights = tf.math.multiply(guided_grads, output)
  if resize_shape is not None:
      weights = tf.reshape(weights, resize_shape)
  cam = tf.reduce_sum(weights, axis=-1).numpy()
  print(output.shape)

  # if resize_shape is None:
  #     weights = tf.reduce_mean(guided_grads, axis=(0, 1))
  # else:
  #     weights = tf.reduce_mean(guided_grads, axis=(0,))
  # cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1).numpy()

  # if resize_shape is not None:
  #     cam = cam.reshape(resize_shape)

  cam = cv2.resize(cam, (image_size, image_size))
  cam = np.maximum(cam, 0)
  # heatmap = (cam - cam.min()) / (cam.max() - cam.min())
  heatmap = cam / cam.max()

  cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
  cam = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 0.5, 0)

  return cam


def grad_cam_list(
  file_name_list, grad_model,
  image_size=256, resize_shape=None,
  pred_index=None
):
  cam_list = []

  for file_name in file_name_list:
    img = tf.keras.preprocessing.image.load_img(
      file_name,
      target_size=(image_size, image_size),
      color_mode = "grayscale")
    img = tf.keras.preprocessing.image.img_to_array(img)

    # print(img.shape)
    cam = grad_cam(img, grad_model, resize_shape=resize_shape, pred_index=pred_index)
    cam_list.append(cam)

  return cam_list
