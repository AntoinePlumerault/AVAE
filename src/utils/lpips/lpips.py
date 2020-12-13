import torch
import numpy as np

from . import models

model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)

def lpips(image_1, image_2):
    image_1 = torch.tensor(np.transpose(image_1.numpy(), [0,3,1,2]))
    image_2 = torch.tensor(np.transpose(image_2.numpy(), [0,3,1,2]))
    distance = model.forward(image_1, image_2)
    distance = np.squeeze(np.array(distance.detach().cpu()))
    return distance