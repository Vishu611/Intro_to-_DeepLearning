from torch.utils.data import Dataset, DataLoader
from PIL import Image
class CustomDataset(Dataset):
  def __init__(self,  transform= None,bg= None, fg_bg= None, fg_bg_mask= None, depth=None):
    self.transform = transform
    self.bg = bg
    self.fg_bg = fg_bg
    self.fg_bg_mask = fg_bg_mask
    self.depth = depth
  
  def __len__(self):
    return len(self.fg_bg)

  def __getitem__(self,index):
    bg_index = index//4000
    bg = Image.open(self.bg[bg_index])
    fg_bg = Image.open(self.fg_bg[index])
    fg_bg_mask = Image.open(self.fg_bg_mask[index])
    depth = Image.open(self.depth[index])

    if self.transform:
      bg = self.transform(bg)
      fg_bg = self.transform(fg_bg)
      fg_bg_mask = self.transform(fg_bg_mask)
      depth = self.transform(depth)
      
    return {'bg' : bg, 'fg_bg' : fg_bg, 'fg_bg_mask' : fg_bg_mask, 'depth' : depth}
