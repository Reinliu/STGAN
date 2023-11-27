from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import glob
import librosa

def create_date_folder(checkpoints_path,name):
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    date = datetime.now()
    day = date.strftime('%d-%m-%Y_')
    path = f'{checkpoints_path}{day}{str(date.hour)}_{name}'
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

#get the number of classes from the number of folders in the audio dir
def get_n_classes(audio_path):
    root, dirs, files = next(os.walk(audio_path))
    n_classes = len(dirs)
    print(f'Found {n_classes} different classes in {audio_path}')
    return n_classes

def min_max_denormalize(normalized_data, original_min, original_max, feature_range=(-1, 1)):
    min_val, max_val = feature_range
    X_std = (normalized_data - min_val) / (max_val - min_val)
    X_denorm = X_std * (original_max - original_min) + original_min
    return X_denorm

def torch_denormalize(normalized_data, original_min, original_max, feature_range=(-1, 1)):
    min_val, max_val = feature_range
    normalized_data = torch.as_tensor(normalized_data).to(original_min.device)
    X_std = (normalized_data - min_val) / (max_val - min_val)
    X_denorm = X_std * (original_max - original_min) + original_min
    return X_denorm


class CustomDataset(Dataset):
    def __init__(self, preprocessed_dir):
        # Load preprocessed data from numpy files
        self.melspecs = np.load(os.path.join(preprocessed_dir, "melspecs.npy"))
        self.loudness = np.load(os.path.join(preprocessed_dir, "loudness.npy"))
        self.labels = np.load(os.path.join(preprocessed_dir, "labels.npy"))
        self.signals = np.load(os.path.join(preprocessed_dir, "signals.npy"))

    def __len__(self):
        return len(self.melspecs)

    def __getitem__(self, idx):
        melspec = torch.tensor(self.melspecs[idx], dtype=torch.float32)
        loudness = torch.tensor(self.loudness[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        signals = torch.tensor(self.signals[idx], dtype=torch.long)
        return melspec, loudness, label, signals
    
def get_dataloader(preprocessed_dir, batch_size=64, shuffle=True, num_workers=0):
    dataset = CustomDataset(preprocessed_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return dataloader


# Vocoder setup
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(melspec, MAX_WAV_VALUE, Vocoder, h, device):
    generator = Vocoder(h).to(device)

    state_dict_g = load_checkpoint('./hifigan/hifigan_vocoder.ckpt', device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()

    with torch.no_grad():
        melspec = torch.FloatTensor(melspec).to(device)
        y_g_hat = generator(melspec)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
    return audio

def inverse_mel_spectrogram(mel_spec, sr, n_fft, n_mels, fmin, fmax):
    # Invert mel scale to linear scale
    mel_basis_inv = np.linalg.pinv(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax))
    return np.dot(mel_basis_inv, mel_spec)

def approximate_loudness_from_melspec(mel_spec, sr, n_fft, fmin, fmax, n_mels):
    print(melspec.min(), melspec.max())
    max_val = np.load('preprocessed/max_val.npy')
    min_val = np.load('preprocessed/min_val.npy')
    melspec = min_max_denormalize(melspec, min_val, max_val)
    print(melspec.min(), melspec.max())
    # Inverse mel-spectrogram to STFT
    spec_approx = inverse_mel_spectrogram(mel_spec, sr, n_fft, n_mels, fmin, fmax)

    # Convert to power spectrum and apply logarithmic scaling
    spec_approx_log = np.log(np.abs(spec_approx)**2 + 1e-7)

    # Apply A-weighting
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    a_weight = librosa.A_weighting(f)
    spec_weighted = spec_approx_log + a_weight.reshape(-1, 1)

    # Average across frequency bands
    loudness_approx = np.mean(spec_weighted, axis=0)

    return loudness_approx

def approximate_loudness_from_batch_melspec(batch_mel_spec, sr, n_fft, fmin, fmax, n_mels):
    max_val = torch.from_numpy(np.load('preprocessed/max_val.npy')).to(batch_mel_spec.device)
    min_val = torch.from_numpy(np.load('preprocessed/min_val.npy')).to(batch_mel_spec.device)
    batch_mel_spec = torch_denormalize(batch_mel_spec, min_val, max_val)
    batch_size = batch_mel_spec.size(0)
    loudness_batch = []

    # Compute the inverted mel basis
    mel_basis_inv = np.linalg.pinv(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax))
    mel_basis_inv = torch.from_numpy(mel_basis_inv).float().to(batch_mel_spec.device)

    for i in range(batch_size):
        # Reshape mel_spec for each item in the batch to 2D: [frequency_bins, time_steps]
        mel_spec = batch_mel_spec[i].squeeze(0)  # Remove the channel dimension, resulting in [400, 64]
        
        # Transpose to get [frequency_bins, time_steps]
        mel_spec = mel_spec.transpose(0, 1)  # Now shape [64, 400]

        # Inverse mel-spectrogram (approximate)
        spec_approx = torch.mm(mel_basis_inv, mel_spec)
        
        # Convert to power spectrum and apply logarithmic scaling
        spec_approx_log = torch.log(spec_approx.pow(2) + 1e-7)

        # Apply A-weighting
        f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        a_weight = librosa.A_weighting(f)
        spec_weighted = spec_approx_log + torch.tensor(a_weight.reshape(-1, 1)).to(spec_approx_log.device)

        # Average across frequency bands to approximate loudness
        loudness_approx = torch.mean(spec_weighted, axis=0)
        loudness_batch.append(loudness_approx)
    return torch.stack(loudness_batch).view(batch_size, 400, 1).float()

# def mel_to_wav(melspec):
#     # Generate melspectrograms
#     melspec = utils.min_max_denormalize(fake_imgs, min_val, max_val)
#     melspec = torch.permute(melspec, (0, 1, 3, 2))
#     melspec = melspec.squeeze(1) 
#     gen_audio = utils.inference(melspec, MAX_WAV_VALUE, Vocoder, h, device)
#     gen_audio = torch.tensor(gen_audio, dtype=torch.float32)
#     target_length = 65536
#     pad_right = target_length - gen_audio.size(1)
#     # Apply padding to the end
#     gen_audio = F.pad(gen_audio, (0, pad_right))  # Only pad the end
#     print(gen_audio.shape)