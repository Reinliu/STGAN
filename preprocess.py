import pathlib
import librosa
import numpy as np
from tqdm import tqdm
import os
import torch
from librosa.filters import mel as librosa_mel_fn
from effortless_config import Config
import json 
   
config_file = 'config.json'
with open(config_file, 'r') as file:
    config = json.load(file)

device = torch.device('cpu')

n_fft=config['n_fft']
n_mels=config['n_mels']
samplerate=16000
hop_size=config['hop_size']
win_size=config['win_size']
fmin=config['fmin']
fmax=config['fmax']
signal_length = 64000
target_length = 65536
MAX_WAV_VALUE = config['MAX_WAV_VALUE']

def load_wav(full_path):
    data, sampling_rate = librosa.load(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def min_max_normalize(data, feature_range=(-1, 1)):
    min_val, max_val = feature_range
    min = np.min(data)
    max = np.max(data)
    X_std = (data - np.min(data)) / (np.max(data) - np.min(data))
    X_scaled = X_std * (max_val - min_val) + min_val
    return X_scaled, min, max


mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, device, center=False):
    y = torch.tensor(y).unsqueeze(0).to(device)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],return_complex=True,
                      center=center, pad_mode='reflect', normalized=False, onesided=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    return spec

def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    S = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weight = librosa.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S

def preprocess_audio(f, n_fft, n_mels, sampling_rate, hop_size, win_size, signal_length, device, oneshot):
    s, _ = librosa.load(f, sr=sampling_rate)
    N = (signal_length - len(s) % signal_length) % signal_length
    s = np.pad(s, (0, N))
    
    if oneshot:
        s = s[..., :signal_length]

    melspecs = mel_spectrogram(s, 
                               n_fft=n_fft, 
                               num_mels=n_mels, 
                               sampling_rate=samplerate, 
                               hop_size=hop_size,
                               win_size=win_size, 
                               fmin=fmin, 
                               fmax=fmax, 
                               device=device)
    loudness = extract_loudness(s, sampling_rate, hop_size)
    label = os.path.basename(os.path.dirname(f))
    return melspecs, loudness, s, label

def main_preprocess(audio_dir, out_dir):
    files = sorted(list(pathlib.Path(audio_dir).rglob("*.wav")))

    all_melspecs = []
    all_loudness = []
    all_signals = []
    all_labels = []
    
    for f in tqdm(files):
        melspecs, loudness, signal, label = preprocess_audio(f, n_fft, n_mels, samplerate, hop_size, win_size, signal_length, device, oneshot=True)
        all_melspecs.append(melspecs.numpy())
        all_loudness.append(loudness)
        all_signals.append(signal)
        all_labels.append(label)
    
    # Convert lists to numpy arrays
    all_melspecs = np.array(all_melspecs)
    all_melspecs, min_val, max_val = min_max_normalize(all_melspecs)
    all_melspecs = np.transpose(all_melspecs, (0,1,3,2))
    all_loudness = np.array(all_loudness)
    all_loudness = np.expand_dims(all_loudness, 2)
    all_signals = np.array(all_signals)
    all_signals = np.pad(all_signals, ((0, 0), (0, target_length - signal_length)), mode='constant', constant_values=0)
    print(all_signals.shape)
    print(all_loudness.shape)
    
    # Create a mapping from labels to integers
    unique_labels = sorted(list(set(all_labels)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    all_labels = np.array([label_to_index[label] for label in all_labels])

    # Save to numpy files
    np.save(os.path.join(out_dir, "melspecs.npy"), all_melspecs)
    np.save(os.path.join(out_dir, "loudness.npy"), all_loudness)
    np.save(os.path.join(out_dir, "labels.npy"), all_labels)
    np.save(os.path.join(out_dir, "signals.npy"), all_signals)
    np.save(os.path.join(out_dir, "min_val.npy"), min_val)
    np.save(os.path.join(out_dir, "max_val.npy"), max_val)

if __name__ == "__main__":
    main_preprocess(
        audio_dir = config['audio_dir'],
        out_dir = config['preprocessed_dir'],
    )