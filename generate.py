# Import libraries
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from models import SpectrogramGenerator
import json
import torch
from hifigan.__init__ import AttrDict
from hifigan.hifigan import Vocoder
from preprocess import preprocess_audio
import utils
import os
import pathlib
from tqdm import tqdm

# Define configure files 
config_file = 'config.json'
with open(config_file, 'r') as file:
    config = json.load(file)
h = AttrDict(config)

# Define parameters
seq_len = 400

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

torch.manual_seed(h.seed)
max_val = np.load('preprocessed/max_val.npy')
min_val = np.load('preprocessed/min_val.npy')


def generate(subfolder, output_base_path, audio_base_path):
    audio_path = pathlib.Path(subfolder)
    files = sorted(list(audio_path.rglob("*.wav")))
    
    # Create a dedicated output subfolder
    output_subfolder = os.path.join(output_base_path, subfolder.name)
    os.makedirs(output_subfolder, exist_ok=True)
    audio_subfolder = os.path.join(audio_base_path, subfolder.name)
    os.makedirs(audio_subfolder, exist_ok=True)
    i=1
    for f in tqdm(files):
        _, loudness, _, _ = preprocess_audio(f, h.n_fft, h.n_mels, h.samplerate, h.hop_size, h.win_size, h.signal_length, device, oneshot=True)
        loudness = torch.tensor(loudness,dtype=torch.float32).to(device)
        loudness = loudness.view(1, loudness.size(0), 1)
        z = torch.randn(1, h.latent_dim).to(device)
        # Duplicate or interpolate loudness and labels if necessary to match num_samples
        
        with torch.no_grad():
            label = torch.tensor([0]).to(device)
            melspec = generator(z, loudness, label).to(device)

        # Rescale spectrogram
        melspec = melspec.detach().cpu().numpy()
        melspec = utils.min_max_denormalize(melspec, min_val, max_val)
        melspec = melspec[0,:,:,:]
        melspec = np.transpose(melspec, (0, 2, 1))

        #Apply vocoder
        waveform = utils.inference(melspec, h.MAX_WAV_VALUE, Vocoder, h, device)
        waveform = waveform[0:64000]
        audio_save_path = os.path.join(audio_subfolder, f'{i}.wav')
        sf.write(audio_save_path, waveform, h.samplerate)
        

        # Save the image in the specific subfolder
        melspec = np.squeeze(melspec)
        output_path = os.path.join(output_subfolder, f'{i}.png')
        plt.imsave(output_path, melspec, cmap='gray', format='png')
        i += 1



# Generator initialization
generator = SpectrogramGenerator(7, seq_len)
generator.to(device)
generator.load_state_dict(torch.load('/home/rein/Documents/GAN-conv1d/saved_model/21-11-2023_21_n_critic_10/generator_10000.pt'))
generator.eval()


if __name__ == "__main__":
    main_folder = '/home/rein/Downloads/development-dataset/'  # Replace with your main directory path
    output_base_path = 'generated/images/images_n_critic_10'  # Base directory for output
    audio_base_path = 'generated/images/audio_n_critic_10'

    for subfolder in pathlib.Path(main_folder).iterdir():
        if subfolder.is_dir():  # Check if it's a directory
            print(f"Processing {subfolder}")
            generate(subfolder, output_base_path, audio_base_path)
