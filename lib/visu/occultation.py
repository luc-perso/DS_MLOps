import numpy as np
import tensorflow as tf
import cv2

# Create function to apply a grey patch on an image
def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
    patched_image = image.copy()
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :] = 127.5

    return patched_image


def occultation(img, model, patch_size=32, sub_samp_for_step=4, pred_index=None):
    sensitivity_map = np.zeros((img.shape[0], img.shape[1]))
    count_map = np.zeros((img.shape[0], img.shape[1]))

    # define prediction
    if pred_index is None:
        predictions = model.predict(np.array([img]), verbose=False)[0]
        pred_index = tf.argmax(predictions)

    # Iterate the patch over the image
    i = 0
    for top_left_x in range(0, img.shape[0], patch_size // sub_samp_for_step):
        for top_left_y in range(0, img.shape[1], patch_size // sub_samp_for_step):
            patched_image = apply_grey_patch(img, top_left_x, top_left_y, patch_size)

            predictions = model(np.expand_dims(patched_image, axis=0), training=False)
            confidence = predictions[:, pred_index]

            # Save confidence for this specific patched image in map
            sensitivity_map[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] \
            = (sensitivity_map[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] \
            * count_map[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] \
            + confidence) \
            / (count_map[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] + 1)

            count_map[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] += 1

        print(str(i) + ',', end='')
        i += 1
    print('')       

          
    heatmap = 1. - (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min() + 1e-6)
    # heatmap = 1. - sensitivity_map / sensitivity_map.max()
    sens = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    occ = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, sens, 0.5, 0)

    return occ


def occultation_list(
  file_name_list, model,
  image_size=256, patch_size=32,
  sub_samp_for_step=4, pred_index=None
):
    occ_list = []

    i = 0
    for file_name in file_name_list:
        img = tf.keras.preprocessing.image.load_img(
          file_name,
          target_size=(image_size, image_size),
          color_mode = "grayscale")
        img = tf.keras.preprocessing.image.img_to_array(img)

        print(str(i) + ':', end='')
        i += 1
        occ = occultation(
            img, model,
            patch_size=patch_size,
            sub_samp_for_step=sub_samp_for_step,
            pred_index=pred_index
        )
        
        occ_list.append(occ)

    return occ_list
