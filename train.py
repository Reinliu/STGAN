import os
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.autograd as autograd
from torchvision.utils import save_image
from models import SpectrogramGenerator, Discriminator
import utils
import auraloss
import json

config_file = 'config.json'
# Open and read the JSON file
with open(config_file, 'r') as file:
    config = json.load(file)

audio_path = config['audio_dir']
os.makedirs("images", exist_ok=True)

n_classes = utils.get_n_classes(audio_path)
n_epochs = config['n_epochs']
batch_size = config['batch_size']
lr = config['lr']
b1 = config['b1']
b2 = config['b2']
n_cpu = config['n_cpu']
latent_dim = config['latent_dim']
n_critic = config['n_critic']
sample_interval = config['sample_interval']
save_freq = config['save_freq']
name = config['name']


cuda = True if torch.cuda.is_available() else False

# Loss weight for gradient penalty
lambda_gp = 10
seq_len = 400

# Initialize generator and discriminator
generator = SpectrogramGenerator(n_classes, seq_len)
discriminator = Discriminator(n_classes)
#audio_discriminator = AudioDiscriminator()

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    #audio_discriminator.cuda()

device = 'cuda:0' if cuda else 'cpu'
print('Using device:', device)


# Configuring HIFIGAN vocoder:
max_val = torch.from_numpy(np.load('preprocessed/max_val.npy'))
min_val = torch.from_numpy(np.load('preprocessed/min_val.npy'))


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
#optimizer_AD = torch.optim.Adam(audio_discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, loudness, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.randn(n_row ** 2, latent_dim, device=device)
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
    dataloader = utils.get_dataloader(out_dir, batch_size, shuffle=True, num_workers=n_cpu)
    checkpoints_path = utils.create_date_folder(save_path, path_name)

    batches_done = 0
    for epoch in tqdm(range(n_epochs)):
        for i, (melspecs, loudness, labels, _) in enumerate(dataloader):
            # Move to GPU if necessary
            real_imgs = melspecs.type(Tensor)
            loudness = loudness.type(Tensor)
            loudness = loudness.permute(0,2,1)
            labels = labels.type(LongTensor)
            # real_audio = signals.type(Tensor)
            # real_audio = real_audio.unsqueeze(1)

            #  Train Discriminator
            optimizer_D.zero_grad()
            #optimizer_AD.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(melspecs.size(0), seq_len, latent_dim).to(device)

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


            optimizer_G.zero_grad()
            # Train the generator every n_critic steps
            if i % n_critic == 0:

                #  Train Generator

                # Generate a batch of images
                fake_imgs = generator(z, loudness, labels)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs, labels)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                if batches_done % sample_interval == 0:
                    #sample_image(opt.n_classes, loudness, batches_done)
                # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                    print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"# [AD loss: %f]"
                    % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                batches_done += n_critic
            
            # if i % (n_critic*10) == 0:
            #     # Generate melspectrograms
            #     melspec = utils.min_max_denormalize(fake_imgs, min_val, max_val)
            #     melspec = torch.permute(melspec, (0, 1, 3, 2))
            #     melspec = melspec.squeeze(1) 
            #     gen_audio = utils.inference(melspec, MAX_WAV_VALUE, Vocoder, h, device)
            #     gen_audio = torch.tensor(gen_audio, dtype=torch.float32, device=device, requires_grad=True)
            #     target_length = 65536
            #     pad_right = target_length - gen_audio.size(1)
            #     fake_audio = F.pad(gen_audio, (0, pad_right)).to(device)  # Only pad the end
            #     fake_audio = fake_audio.unsqueeze(1)


            #     # Audio aural loss stuff
            #     loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
            #     fft_sizes=[1024, 2048, 8192],
            #     hop_sizes=[256, 512, 2048],
            #     win_lengths=[1024, 2048, 8192],
            #     scale="mel",
            #     n_bins=128,
            #     sample_rate=16000,
            #     perceptual_weighting=True,)
            #     ad_loss = loss_fn(fake_audio, real_audio)

            #     # # Real images
            #     # real_audio_validity = audio_discriminator(real_audio, labels)
            #     # # Fake images
            #     # fake_audio_validity = audio_discriminator(fake_audio, labels)
            #     # # Adversarial loss
            #     # ad_loss = -torch.mean(real_audio_validity) + torch.mean(fake_audio_validity)

            #     ad_loss.backward()
            #     optimizer_AD.step()
            
        if epoch >= save_freq and (epoch-save_freq) % save_freq == 0:
            torch.save(generator.state_dict(), f'{checkpoints_path}/generator_{epoch}.pt')

if __name__ == "__main__":
    preprocessed_dir = config['preprocessed_dir']  # Adjust this to the directory where you saved the data.
    save_path = config['save_path']
    
    train(preprocessed_dir, save_path, name)