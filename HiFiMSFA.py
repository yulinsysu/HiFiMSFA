import torch
import torch.nn as nn
from kornia.filters import GaussianBlur2d, SpatialGradient, Laplacian


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False), nn.LeakyReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)

class SFExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.blur0 = GaussianBlur2d(kernel_size=(3,3), sigma=(0.2,0.2))
        self.blur1 = GaussianBlur2d(kernel_size=(5,5), sigma=(0.5,0.5))
        self.blur2 = GaussianBlur2d(kernel_size=(int(6*1+1),int(6*1+1)), sigma=(1,1))
        self.blur3 = GaussianBlur2d(kernel_size=(int(6*2+1),int(6*2+1)), sigma=(2,2))
        self.blur4 = GaussianBlur2d(kernel_size=(int(6*3+1),int(6*3+1)), sigma=(3,3))
        self.blur5 = GaussianBlur2d(kernel_size=(int(6*4+1),int(6*4+1)), sigma=(4,4))
        self.laplacian = Laplacian(kernel_size=5)
        self.sgrad = SpatialGradient()
    def forward(self, image):
        B, _, H, W = image.shape
        img0 = self.blur0(image)
        img1 = self.blur1(image)
        img2 = self.blur2(image)
        img3 = self.blur3(image)
        img4 = self.blur4(image)
        img5 = self.blur5(image)
        out = torch.cat([image, (img0-img1).abs(), (img1-img2).abs(), (img2-img3).abs(), (img3-img4).abs(), (img4-img5).abs(), self.laplacian(image).abs(), self.sgrad(image).abs().reshape(B,-1,H,W)], dim=1)
        return out

class MSFABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicBlock(3, 16, stride=1),
            BasicBlock(16, 32, stride=2),
        )
        self.conv2 = nn.Sequential(
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 64, stride=2)
        )
        self.conv3 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 128, stride=2)
        )
        self.img_down = nn.AvgPool2d((2,2))
        self.sfe = SFExtractor()
        self.sf_att1 = nn.Sequential(
            nn.Conv2d(27+32, 32, 3, stride=1, padding=1), nn.Sigmoid()
        )
        self.sf_att2 = nn.Sequential(
            nn.Conv2d(27+64, 64, 3, stride=1, padding=1), nn.Sigmoid()
        )
        self.sf_att3 = nn.Sequential(
            nn.Conv2d(27+128, 128, 3, stride=1, padding=1), nn.Sigmoid()
        )
    def forward(self, image):
        x = self.conv1(image)
        image = self.img_down(image)
        x = self.sf_att1(torch.cat([x,self.sfe(image)],dim=1)) * x
        x = self.conv2(x)
        image = self.img_down(image)
        x = self.sf_att2(torch.cat([x,self.sfe(image)],dim=1)) * x
        x = self.conv3(x)
        image = self.img_down(image)
        x = self.sf_att3(torch.cat([x,self.sfe(image)],dim=1)) * x
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = MSFABlock()
        self.embedder = nn.Sequential(
            nn.Conv2d(129, 128, 1, stride=1, padding=0), nn.LeakyReLU(inplace=True),
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 128, stride=1),
            nn.Conv2d(128, 192, 1, stride=1, padding=0), nn.LeakyReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=8)
        )
        self.residual = nn.Sequential(
            BasicBlock(3+3, 32, stride=1),
            BasicBlock(32, 32, stride=1),
            nn.Conv2d(32, 3, kernel_size=1)
        )
    def forward(self, image, msg):
        x = self.block(image)
        x = self.embedder(torch.cat([msg.reshape(-1,1,8,8).repeat(1,1,2,2), x], dim=1))
        return image + self.residual(torch.cat([image, x], dim=1))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            MSFABlock(),
            BasicBlock(128, 128, stride=1),
            nn.Conv2d(128, 1, 3, stride=1, padding=1),
            nn.PixelUnshuffle(downscale_factor=8)
        )
    def forward(self, image):
        return self.decoder(image).mean(dim=[2,3])

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.out = nn.Linear(32, 1)
    def forward(self, image):
        x = self.net(image)
        x = x.reshape(x.shape[0], -1)
        x = self.out(x)
        return x


class ASHL():
    def __init__(self):
        self.gaussblur = GaussianBlur2d((11,11), (5,5))
    
    def __call__(self, image, extract_msg, origin_msg):
        blurred = self.gaussblur(image)
        T = (image-blurred).abs()
        T = T.max(dim=1, keepdim=True)[0]
        t = nn.functional.avg_pool2d(T, 8)
        r = nn.functional.pixel_unshuffle(t, downscale_factor=8).mean(dim=[2,3]) + 0.5
        loss = (1-origin_msg) * nn.functional.relu(extract_msg-0.5+r) + origin_msg * nn.functional.relu(-extract_msg+0.5+r)
        return (loss ** 2).mean()
