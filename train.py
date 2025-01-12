import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from HiFiMSFA import Encoder, Decoder, Discriminator, ASHL

class DataSet(Dataset):
    def __init__(self, image_folder):
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None: m.bias.data.zero_()

def train():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'mirflickr/train/'
    args.device = 'cuda'
    args.epochs = 40
    args.lr = 0.0005
    args.batch_size = 16
    args.msg_size = 64
    args.log_step = 100
    args.save_step = 5000
    
    trainloader = torch.utils.data.DataLoader(DataSet(args.dataset), batch_size=args.batch_size, shuffle=True, drop_last=True)

    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    discriminator = Discriminator().to(args.device)
    label_real = torch.ones(args.batch_size,1).to(args.device)
    label_fake = torch.zeros(args.batch_size,1).to(args.device)

    encoder.apply(weight_init)
    decoder.apply(weight_init)
    discriminator.apply(weight_init)

    msgloss = ASHL()
    imgloss = nn.MSELoss()
    advloss = nn.BCEWithLogitsLoss()

    optimizer_coder = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr)
    optimizer_discr = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    print("start training.")
    lambda1 = 1
    lambda2 = 1
    lambda3 = 0.0001
    lambda1_decay = 1
    step = 0
    for epoch in range(args.epochs):
        for data in trainloader:
            cover = data.to(args.device)
            origin_msg = torch.randint(0, 2, (args.batch_size,args.msg_size)).float().to(args.device)

            stego = encoder(cover, origin_msg).clamp(-1,1)
            noised = stego + torch.randn(*cover.shape).to(args.device)*0.05*2
            extract_msg = decoder(noised)

            # train the discriminator
            optimizer_discr.zero_grad()
            d_loss = advloss(discriminator(cover).mean(dim=1,keepdim=True), label_real) + advloss(discriminator(stego.detach()).mean(dim=1,keepdim=True), label_fake)
            d_loss.backward()
            optimizer_discr.step()
            
            # train the encoder and decoder
            msg_loss = msgloss(cover, extract_msg, origin_msg)
            img_loss = imgloss(stego, cover)
            adv_loss = advloss(discriminator(stego).mean(dim=1,keepdim=True), label_real)

            lambda1_decay = 10 ** (np.clip((step-1000)/(10000-1000), 0, 1) * 3)
            loss = lambda1/lambda1_decay*msg_loss + lambda2*img_loss + lambda3*adv_loss
            optimizer_coder.zero_grad()
            loss.backward()
            optimizer_coder.step()
            
            step += 1
            if step % args.log_step == 0:
                accu = ((extract_msg >= 0.5).eq(origin_msg >= 0.5).sum().float() / origin_msg.numel()).item()
                psnr = 10 * torch.log10(4 / torch.mean((cover-stego)**2)).item()
                print('step:', step, 'accu:', accu, 'psnr:', psnr)
            if step % args.save_step == 0:
                torch.save(encoder.state_dict(), 'checkpoints/HiFiMSFA-encoder.pth')
                torch.save(decoder.state_dict(), 'checkpoints/HiFiMSFA-decoder.pth')
                torch.save(discriminator.state_dict(), 'checkpoints/HiFiMSFA-discriminator.pth')
        if epoch == 25:
            for param_group in optimizer_coder.param_groups:
               param_group['lr'] = 0.0002
            for param_group in optimizer_discr.param_groups:
               param_group['lr'] = 0.0002
        print('epoch:', epoch)


if __name__ == '__main__':
    train()