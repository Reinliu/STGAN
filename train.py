import os
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.autograd as autograd
from torchvision.utils import save_image
from models import SpectrogramGenerator, Discriminator
from hifigan.__init__ import AttrDict
import utils
import auraloss
import json

config_file = 'config.json'
with open(config_file, 'r') as file:
    config = json.load(file)
h = AttrDict(config)

os.makedirs("images", exist_ok=True)
n_classes = utils.get_n_classes(h.audio_dir)

cuda = True if torch.cuda.is_available() else False

# Loss weight for gradient penalty
lambda_gp = 10
seq_len = 400

# Initialize generator and discriminator
generator = SpectrogramGenerator(n_classes, seq_len)
discriminator = Discriminator(n_classes)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

device = 'cuda:0' if cuda else 'cpu'
print('Using device:', device)


# Configuring HIFIGAN vocoder:
max_val = torch.from_numpy(np.load('preprocessed/max_val.npy'))
min_val = torch.from_numpy(np.load('preprocessed/min_val.npy'))


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=h.lr, betas=(h.b1, h.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=h.lr, betas=(h.b1, h.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, loudness, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.randn(n_row ** 2, seq_len, h.latent_dim, device=device)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    with torch.no_grad():
        labels = LongTensor(labels)
        gen_imgs = generator(z, loudness, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(out_dir, save_path, path_name):
    dataloader = utils.get_dataloader(out_dir, h.batch_size, shuffle=True, num_workers=h.n_cpu)
    checkpoints_path = utils.create_date_folder(save_path, path_name)

    batches_done = 0
    mse_loss_fn = torch.nn.MSELoss()
    for epoch in tqdm(range(h.n_epochs)):
        for i, (melspecs, loudness, labels, _) in enumerate(dataloader):
            # Move to GPU if necessary
            real_imgs = melspecs.type(Tensor)
            loudness = loudness.type(Tensor)
            labels = labels.type(LongTensor)
            # real_audio = signals.type(Tensor)
            # real_audio = real_audio.unsqueeze(1)

            #  Train Discriminator
            optimizer_D.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(melspecs.size(0), h.latent_dim).to(device)

            # Generate a batch of images
            fake_imgs = generator(z, loudness, labels)

            # Real images
            real_validity = discriminator(real_imgs, labels)
            # Fake images
            fake_validity = discriminator(fake_imgs, labels)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                                discriminator, real_imgs.data, fake_imgs.data,
                                labels.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            # Train the generator every n_critic steps
            optimizer_G.zero_grad()
            if i % h.n_critic == 0:

                # Generate a batch of images
                fake_imgs = generator(z, loudness, labels)
                # Train on fake images
                fake_validity = discriminator(fake_imgs, labels)
                g_loss = -torch.mean(fake_validity)
                fake_loudness = utils.approximate_loudness_from_batch_melspec(fake_imgs, h.samplerate, h.n_fft, h.fmin, h.fmax, h.n_mels)
                mse_loss = h.mse_weight * mse_loss_fn(fake_loudness, loudness)
                g_loss = g_loss + mse_loss
                g_loss.backward()
                optimizer_G.step()

                if batches_done % h.sample_interval == 0:
                    # sample_image(n_classes, loudness, batches_done)
                    # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                    print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"# [AD loss: %f]"
                    % (epoch, h.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                batches_done += h.n_critic

            
        if epoch >= h.save_freq and (epoch-h.save_freq) % h.save_freq == 0:
            torch.save(generator.state_dict(), f'{checkpoints_path}/generator_{epoch}.pt')

if __name__ == "__main__":
    
    train(h.preprocessed_dir, h.save_path, h.save_name)