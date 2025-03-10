import torch
from dataset import PhotoMonetDataset
import sys
import os
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc_P, disc_M, gen_P, gen_M, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop= tqdm(loader,leave=True)    # progress bar
    for idx, (monet,photo) in enumerate(loop):
        print(f"Entering iteration {idx}")
        photo = photo.to(config.DEVICE)
        monet = monet.to(config.DEVICE)
 # Train Discriminators H and Z.
        with torch.amp.autocast('cuda'):
            fake_Photo= gen_P(monet)
            D_P_real= disc_P(photo)
            D_P_fake= disc_P(fake_Photo.detach())
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_fake_loss+D_P_real_loss

            fake_Monet = gen_M(photo)
            D_M_real = disc_P(monet)
            D_M_fake = disc_P(fake_Monet.detach())
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_fake_loss + D_M_real_loss

            D_loss= (D_P_loss+D_M_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

    # Train generators P and M
        with ((torch.amp.autocast('cuda'))):
            # Adverserial loss
            D_P_fake = disc_P(fake_Photo)
            D_M_fake=  disc_M(fake_Monet)
            Loss_G_M= mse(D_M_fake, torch.ones_like(D_M_fake))
            Loss_G_P= mse(D_P_fake, torch.ones_like(D_P_fake))
            # Cycle loss
            cycle_monet= gen_M(fake_Photo)
            cycle_photo= gen_P(fake_Monet)
            cycle_monet_loss= l1(cycle_monet, monet)
            cycle_photo_loss= l1(cycle_photo,photo)
            # Identitiy loss
            identity_photo= gen_P(photo)
            identity_monet= gen_M(monet)
            identity_monet_loss= l1(identity_photo,photo)
            identity_photo_loss= l1(identity_monet,monet)
            # Add all together
            G_loss = (Loss_G_M+Loss_G_P
            + cycle_monet_loss* config.LAMBDA_CYCLE
            + cycle_photo_loss* config.LAMBDA_CYCLE
            + identity_monet_loss * config.LAMBDA_IDENTITY
            + identity_photo_loss * config.LAMBDA_IDENTITY)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 20 == 0:
            print("hello")
            save_image(fake_Photo * 0.5 + 0.5, f"saved_images/photo_{idx}.png")
            save_image(fake_Monet * 0.5 + 0.5, f"saved_images/monet_{idx}.png")

def main():
    disc_P= Discriminator(in_channels=3).to(config.DEVICE)
    disc_M= Discriminator(in_channels=3).to(config.DEVICE)
    gen_P= Generator(img_channels=3, num_residuals=9). to (config.DEVICE)
    gen_M= Generator(img_channels=3, num_residuals=9). to (config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_P.parameters()) + list(disc_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_P.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1= nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_P,
            gen_P,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_M,
            gen_M,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_P,
            disc_P,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_M,
            disc_M,
            opt_disc,
            config.LEARNING_RATE,
        )
# These checkpoint files allow the training process to resume from where it left off, without starting over from scratch.
    dataset = PhotoMonetDataset(
        root_photo=config.TRAIN_DIR + "/Photo",
        root_monet=config.TRAIN_DIR + "/Monet",
        transform=config.transforms,
    )
    # val_dataset = PhotoMonetDataset(
    #     root_photo="cyclegan_test/photo1",
    #     root_monet="cyclegan_test/monet1",
    #     transform=config.transforms,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    # )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_P,
            disc_M,
            gen_P,
            gen_M,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_P, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_M, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_P, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_M, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
    main()



