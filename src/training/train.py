# src/training/train.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from data.dataset import MonoStereoDataset
from models.unet import UNetAudio

def train_model():
    # Load configurations
    config = Config()

    # Print CUDA availability and device info
    print("Is CUDA available?", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current CUDA device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available. Using CPU.")

    # Print the device being used
    print(f"Using device: {config.DEVICE}")

    # Initialize model
    model = UNetAudio().to(config.DEVICE)

    # Print model device
    for param in model.parameters():
        print(f"Model parameter device: {param.device}")
        break  # Only need to print for one parameter

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Load dataset
    dataset = MonoStereoDataset(
        mono_dir=config.MONO_DIR,
        stereo_dir=config.STEREO_DIR,
        sample_rate=config.SAMPLE_RATE,
        chunk_size=config.CHUNK_SIZE
    )
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        for mono, stereo in progress_bar:
            mono = mono.to(config.DEVICE)
            stereo = stereo.to(config.DEVICE)

            # Forward pass
            outputs = model(mono)

            # Compute loss
            loss = criterion(outputs, stereo)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], Average Loss: {avg_loss:.6f}')

    # Save the final model
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f'Model training completed and saved to {config.MODEL_SAVE_PATH}')

if __name__ == '__main__':
    train_model()
