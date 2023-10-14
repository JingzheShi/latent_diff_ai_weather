import torch
import torch.nn as nn
import torch.optim as optim
from models.ddpm import DiffusionWrapper

class DiffusionProcess(nn.Module):
    def __init__(self, img_size, num_steps, denoising_network):
        super(DiffusionProcess, self).__init__()
        self.img_size = img_size
        self.num_steps = num_steps
        self.denoising_net = denoising_network

        # Betas for the diffusion process, which can be set as learnable or predetermined
        self.betas = nn.Parameter(torch.linspace(0.0001, 0.02, steps=num_steps), requires_grad=False)

    def forward(self, x, reverse=False):
        """
        If reverse is False: This function will transform the input image into noise.
        If reverse is True: This function will transform the noise back to an image.
        """
        if not reverse:
            for i in range(self.num_steps):
                noise = torch.randn_like(x) * torch.sqrt(self.betas[i])
                x = x + noise - self.denoising_net(x, i) * self.betas[i]
        else:
            for i in reversed(range(self.num_steps)):
                x = (x + self.denoising_net(x, i) * self.betas[i]) / (1.0 - self.betas[i])

        return x

# Example Usage:
img_size = (3, 64, 64)
num_steps = 100
denoising_network = Unet(feature_num=1, unet_layer_num=4)
diffusion_model = DiffusionProcess(img_size, num_steps, denoising_network)

# For a given input image
input_image = torch.randn(1, *img_size)
noised = diffusion_model(input_image, reverse=False)
reconstructed = diffusion_model(noised, reverse=True)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(diffusion_model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):  # Assuming the dataloader returns images and labels
        data = data.to(device)  # Move data to the device (CPU/GPU)
        
        optimizer.zero_grad()  # Zero out gradients
        
        # Corrupt the image with the diffusion process
        noised = diffusion_model(data, reverse=False)
        
        # Attempt to recover the original image
        reconstructed = diffusion_model(noised, reverse=True)
        
        # Compute the loss between the original and the reconstructed image
        loss = criterion(reconstructed, data)
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Optionally, print the loss every few batches
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx} | Loss: {loss.item():.6f}")

print("Training complete!")