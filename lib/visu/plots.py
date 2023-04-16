import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def plot_image_list(
    image_list, file_name_liste,
    nrows, ncols,
    fig_title, fig_file_name,
    output_path
):
  fig = plt.figure(figsize=(12, 12))
  fig.suptitle(fig_title)
  ax = fig.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
  for i, (image_list, file_name) in enumerate(zip(image_list, file_name_liste)):
    axe = ax.flatten()[i]

    img = cv2.cvtColor(image_list, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    axe.imshow(im_pil)
    f_name = os.path.basename(file_name)[:10]
    # print(f_name)
    axe.set_title(f_name)

  plt.savefig(os.path.join(output_path , fig_file_name + '.png'), format='png')
  plt.show()
