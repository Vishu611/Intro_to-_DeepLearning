#Albumentation transforms
from torchvision import transforms
import torch
def AlbumentationTransforms():
    mean = torch.tensor([0.65830478, 0.61511271, 0.5740604])*255
    std = torch.tensor([0.24408717, 0.2542491, 0.26870159])*255
  
    train_transform = transforms.Compose([
                      transforms.Resize((244,244)),
                     # transforms.Resize((64,64)),
                      transforms.ColorJitter(brightness=0.05,contrast=0.05,saturation=0.05,hue=0.05),
                      transforms.ToTensor()
    ])
    return train_transform