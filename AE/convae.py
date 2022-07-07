import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.autoencoder = nn.Sequential(
            ## Encoder
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            ## Decoder
            nn.Conv2d(12, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, imgs):
        x = self.autoencoder(imgs)
        return x
