import os

import torch
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from PIL import Image

from utils.models import Encoder, Decoder, adaptive_instance_normalization


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path, size=512):
    image = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    return transform(image).unsqueeze(0).to(device)


def train():

    print("Training device:", device)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    encoder.eval()
    decoder.train()

    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

    content_dir = "../Demo_IO_Images/content"
    style_dir = "../Demo_IO_Images/style"

    content_images = []
    style_images = []

    if os.path.exists(content_dir):
        content_images = [
            os.path.join(content_dir, file)
            for file in os.listdir(content_dir)
        ]

    if os.path.exists(style_dir):
        style_images = [
            os.path.join(style_dir, file)
            for file in os.listdir(style_dir)
        ]

    if not content_images or not style_images:
        print("Add content and style images before training.")
        return

    for epoch in range(10):

        total_loss = 0.0

        for content_path, style_path in zip(
            content_images,
            style_images
        ):

            content = load_image(content_path)
            style = load_image(style_path)

            with torch.no_grad():
                content_feature = encoder(content)
                style_feature = encoder(style)

                target_feature = adaptive_instance_normalization(
                    content_feature,
                    style_feature
                )

            output = decoder(target_feature)

            output_feature = encoder(output)

            loss = F.mse_loss(
                output_feature,
                target_feature
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/10 "
            f"- Loss: {total_loss:.4f}"
        )

    os.makedirs("../experiment/final_exp", exist_ok=True)

    torch.save(
        decoder.state_dict(),
        "../experiment/final_exp/decoder_final.pth"
    )

    print("Decoder saved successfully.")


if __name__ == "__main__":
    train()