import torch
from torch import nn

class TinyVGG(nn.Module):
    def __init__(self,input_shape:int,output_shape:int,hidden_shape:int)-> None :
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_shape,hidden_shape,3,1,0),
            nn.ReLU(),
            nn.Conv2d(hidden_shape,hidden_shape,3,1,0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block2=nn.Sequential(
            nn.Conv2d(hidden_shape,hidden_shape,3,padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_shape,hidden_shape,3,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_shape*13*13,output_shape)
        )
    def forward(self,x:torch.Tensor):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

