from torch.utils.data import random_split
import torch
def split(train_dataset,train_split=70):
  train_len = len(train_dataset)*train_split//100
  test_len = len(train_dataset) - train_len 
  print("Total dataset = ",len(train_dataset))
  torch.manual_seed(0)
  train_set, val_set = random_split(train_dataset, [train_len, test_len])  
  print("Train set = ",len(train_set))
  print("Test set = ",len(val_set))
  return train_set, val_set