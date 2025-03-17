import torch
from torchvision import transforms
from PIL import Image
import config
from generator_model import Generator


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

device = config.DEVICE  # Use GPU if available

gen_P = Generator(img_channels=3, num_residuals=9).to(device)  # Monet → Photo
gen_M = Generator(img_channels=3, num_residuals=9).to(device)  # Photo → Monet

# Load pre-trained weights
checkpoint= torch.load(config.CHECKPOINT_GEN_P, map_location=device)
gen_P.load_state_dict(checkpoint["state_dict"])

checkpoint= torch.load(config.CHECKPOINT_GEN_M, map_location=device)
gen_M.load_state_dict(checkpoint["state_dict"])


gen_P.eval()
gen_M.eval()


def generate_image(input_path, output_path, model,device):
    image = Image.open(input_path)
    img_transformed = transform(image)  # Apply the same transformation
    img_transformed = img_transformed.unsqueeze(0).to(device)

    with torch.no_grad():
        fake_image = model(img_transformed)  # Forward pass through generator
    fake_image = fake_image.squeeze(0).cpu()  # Remove batch dimension

    # Convert back to image format
    save_image = transforms.ToPILImage()(fake_image * 0.5 + 0.5)  # Denormalize
    save_image.save(output_path)
    print('hello')

generate_image('Test_input_2/Monet/bbc5ac4564.jpg', 'Test_output_2/Monet/bbc5ac4564_monet.jpg', gen_M,config.DEVICE)
