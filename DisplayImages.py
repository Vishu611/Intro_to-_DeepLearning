import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms,utils
def display_images(tensors, figsize=(10,10), *args, **kwargs):
  try:
    tensors = tensors.detach().cpu()
  except: 
    pass
  grid_tensors= torchvision.utils.make_grid(tensors,*args, **kwargs)
  grid_image= grid_tensors.permute(1,2,0)
  plt.figure(figsize=figsize)
  plt.imshow(grid_image)
  plt.xticks([])
  plt.yticks([])
  plt.show()