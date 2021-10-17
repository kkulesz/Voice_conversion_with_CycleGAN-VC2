import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from src.utils.utils import Utils
from sandpit.sandpit_utils import sandpit_datasets_dir_path, sandpit_output_dir_path


class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(in_features=noise_dim, out_features=256),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=256, out_features=output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.sequential(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.LeakyReLU(.1),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequential(x)


################################################
lr = 1e-3
batch_size = 32
num_of_epochs = 50
device = Utils.get_device()

noise_dimension = 64
image_dimension = 28 * 28 * 1

loss_fn = nn.BCELoss()
################################################


def disc_loss_fn(d_real, d_fake):
    d_loss_real = loss_fn(d_real, torch.ones_like(d_real))
    d_loss_fake = loss_fn(d_fake, torch.zeros_like(d_fake))

    return (d_loss_real + d_loss_fake) / 2


def gen_loss_fn(d_fake):
    return loss_fn(d_fake, torch.ones_like(d_fake))


def train(gen, gen_optimizer, disc, disc_optimizer, dataloader):
    fixed_noise = torch.randn((batch_size, noise_dimension)).to(device)

    for epoch_idx in range(num_of_epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.reshape(-1, image_dimension).to(device)
            noise = torch.randn(batch_size, noise_dimension).to(device)

            # Training Discriminator
            fake = gen(noise)
            d_real = disc(real).reshape(-1)
            d_fake = disc(fake).reshape(-1)

            d_loss = disc_loss_fn(d_real=d_real, d_fake=d_fake)

            disc.zero_grad()
            d_loss.backward(retain_graph=True)  # retain_graph=True because we want to use it later for generator
            disc_optimizer.step()

            # Training Generator
            d_fake_for_gen = disc(fake).reshape(-1)
            g_loss = gen_loss_fn(d_fake_for_gen)

            gen.zero_grad()
            g_loss.backward()
            gen_optimizer.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch_idx}/{num_of_epochs}] Batch {batch_idx}/{len(dataloader)} \
                          Discriminator loss: {d_loss:.4f}, Generator loss: {g_loss:.4f}"
                )
                with torch.no_grad():
                    to_save = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    img_grid_to_save = torchvision.utils.make_grid(to_save, normalize=True)

                    prefix = ''  # so images are not overwritten - ugly but works
                    file_to_save_path = os.path.join(sandpit_output_dir_path, prefix + str(epoch_idx) + '.png')

                    torchvision.utils.save_image(img_grid_to_save,
                                                 file_to_save_path)


if __name__ == '__main__':
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(0.5, 0.5, inplace=False)]
    )
    mnist_dataset = torchvision.datasets.MNIST(root=sandpit_datasets_dir_path, transform=transforms, download=True)
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(noise_dim=noise_dimension, output_dim=image_dimension).to(device)
    discriminator = Discriminator(input_dim=image_dimension).to(device)

    generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=lr)
    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=lr)

    train(gen=generator, gen_optimizer=generator_optimizer,
          disc=discriminator, disc_optimizer=discriminator_optimizer,
          dataloader=mnist_dataloader)
