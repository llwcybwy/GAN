import torch
from dataset import PhotoMonetDataset
import CMMD
import sys
import os
from torchvision import transforms
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def validate_fn(disc_P, disc_M, gen_P, gen_M, loader, l1, mse, epoch):
    # Set models to evaluation mode
    disc_P.eval()
    disc_M.eval()
    gen_P.eval()
    gen_M.eval()

    total_D_loss = 0.0
    total_G_loss = 0.0
    num_batches = len(loader)

    # Disable gradient computations for validation
    with torch.no_grad():
        loop = tqdm(loader, leave=True)
        for idx, (monet, photo) in enumerate(loop):
            photo = photo.to(config.DEVICE)
            monet = monet.to(config.DEVICE)
            # --------------------
            #  Discriminator Loss
            # --------------------
            with torch.amp.autocast('cuda'):
                fake_Photo = gen_P(monet)
                D_P_real = disc_P(photo)
                D_P_fake = disc_P(fake_Photo)
                D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
                D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
                D_P_loss = D_P_real_loss + D_P_fake_loss

                fake_Monet = gen_M(photo)
                D_M_real = disc_M(monet)
                D_M_fake = disc_M(fake_Monet)
                D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
                D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
                D_M_loss = D_M_real_loss + D_M_fake_loss

                D_loss = (D_P_loss + D_M_loss) / 2

            # --------------------
            #  Generator Loss
            # --------------------
            with torch.amp.autocast('cuda'):
                # Re-compute for generators (if needed for validation metrics)
                D_P_fake = disc_P(fake_Photo)
                D_M_fake = disc_M(fake_Monet)
                Loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))
                Loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))

                # Cycle consistency
                cycle_monet = gen_M(fake_Photo)
                cycle_photo = gen_P(fake_Monet)
                cycle_monet_loss = l1(cycle_monet, monet)
                cycle_photo_loss = l1(cycle_photo, photo)

                # Identity loss
                identity_photo = gen_P(photo)
                identity_monet = gen_M(monet)
                identity_monet_loss = l1(identity_photo, photo)
                identity_photo_loss = l1(identity_monet, monet)

                G_loss = (Loss_G_P + Loss_G_M +
                          cycle_monet_loss * config.LAMBDA_CYCLE +
                          cycle_photo_loss * config.LAMBDA_CYCLE +
                          identity_monet_loss * config.LAMBDA_IDENTITY +
                          identity_photo_loss * config.LAMBDA_IDENTITY)
            # --------------------
            #  CMMD
            # --------------------
            with torch.no_grad():
                fake_Monet = fake_Monet.squeeze(0).cpu()  # Remove batch dimension
                fake_Photo = fake_Photo.squeeze(0).cpu()
                # Convert back to image format
                save_image_Monet = transforms.ToPILImage()(fake_Monet * 0.5 + 0.5)  # Denormalize
                save_image_Photo = transforms.ToPILImage()(fake_Photo * 0.5 + 0.5)
                save_image_Monet.save(f'Output/Monet/{epoch}/{idx}.jpg')
                save_image_Photo.save(f'Output/Photo/{epoch}/{idx}.jpg')
            total_D_loss += D_loss.item()
            total_G_loss += G_loss.item()

    avg_D_loss = total_D_loss / num_batches
    avg_G_loss = total_G_loss / num_batches
    # print(f"Epoch {epoch} | Validation D Loss: {avg_D_loss:.4f}, G Loss: {avg_G_loss:.4f}")

    # Set models back to training mode
    disc_P.train()
    disc_M.train()
    gen_P.train()
    gen_M.train()

    return avg_D_loss, avg_G_loss
def train_fn(disc_P, disc_M, gen_P, gen_M, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop= tqdm(loader,leave=True)    # progress bar
    for idx, (monet,photo) in enumerate(loop):
        print(f"Entering iteration {idx}")
        photo = photo.to(config.DEVICE)
        monet = monet.to(config.DEVICE)
 # Train Discriminators P and M.
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
        if idx % 1000 == 0:
            print("hello")
            save_image(fake_Photo * 0.5 + 0.5, f"saved_images_4/photo_{idx}.png")
            save_image(fake_Monet * 0.5 + 0.5, f"saved_images_4/monet_{idx}.png")
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
    # These checkpoint files allow the training process to resume from where it left off, without starting over from scratch.
    if config.LOAD_MODEL:
        load_checkpoint('Output/Training_parameters/3/genh.pth.tar'
            ,
            gen_P,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint('Output/Training_parameters/3/genz.pth.tar'
            ,
            gen_M,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint('Output/Training_parameters/3/critich.pth.tar'
            ,
            disc_P,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint('Output/Training_parameters/3/criticz.pth.tar'
            ,
            disc_M,
            opt_disc,
            config.LEARNING_RATE,
        )

    train_dataset = PhotoMonetDataset(
        root_photo=config.TRAIN_DIR + "/Photo",
        root_monet=config.TRAIN_DIR + "/Monet",
        transform=config.transforms,
    )
    val_dataset = PhotoMonetDataset(
         root_photo=config.VAL_DIR + "/Photo",
         root_monet=config.VAL_DIR + "/Monet",
         transform=config.transforms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )


    # loader = DataLoader(
    #     dataset,
    #     batch_size=config.BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=config.NUM_WORKERS,
    #     pin_memory=True,
    # )
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')

    G_loss = []
    D_loss = []
    CMMD_Monet = []
    CMMD_Photo = []

    for epoch in range(4,7):
        print(f"Epoch {epoch}:")
        print("Training...")
        train_fn(
            disc_P,
            disc_M,
            gen_P,
            gen_M,
            train_loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )
        print("Validation...")
        new_g_loss,new_d_loss = validate_fn(
            disc_P,
            disc_M,
            gen_P,
            gen_M,
            val_loader,
            L1,
            mse,
            epoch,
        )
        new_CMMD_Monet= CMMD.compute_cmmd('Data/val/Monet', f'Output/Monet/{epoch}')
        new_CMMD_Photo= CMMD.compute_cmmd('Data/val/Photo', f'Output/Photo/{epoch}')
        G_loss.append(new_g_loss)
        D_loss.append(new_d_loss)
        CMMD_Monet.append(new_CMMD_Monet)
        CMMD_Photo.append(new_CMMD_Photo)
        print("G loss: ", G_loss)
        print("D loss:", D_loss)
        print("CMMD Monet:", CMMD_Monet)
        print("CMMD Photo:", CMMD_Photo)


        if config.SAVE_MODEL:
            save_checkpoint(gen_P, opt_gen, filename= f'Output/Training_parameters/{epoch}/'+ config.CHECKPOINT_GEN_P)
            save_checkpoint(gen_M, opt_gen, filename= f'Output/Training_parameters/{epoch}/'+ config.CHECKPOINT_GEN_M)
            save_checkpoint(disc_P, opt_disc, filename= f'Output/Training_parameters/{epoch}/'+config.CHECKPOINT_CRITIC_P)
            save_checkpoint(disc_M, opt_disc, filename= f'Output/Training_parameters/{epoch}/'+ config.CHECKPOINT_CRITIC_M)

    print("G loss: ", G_loss)
    print("D loss:", D_loss)
    print("CMMD Monet:", CMMD_Monet)
    print("CMMD Photo:", CMMD_Photo)

if __name__ == "__main__":
    main()



