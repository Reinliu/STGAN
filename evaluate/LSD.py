import os
import numpy as np
import librosa

def calculate_spectral(directory1, directory2, sampling_rate=16000):
    spectral_convergences = []
    log_spectral_distances = []

    # Create a mapping of file names from directory2 to directory1
    file_map = {str(int(os.path.splitext(file)[0])): file for file in os.listdir(directory2)}

    for file in os.listdir(directory1):
        file_base = str(int(os.path.splitext(file)[0]))

        if file_base in file_map:
            # Construct file paths
            path1 = os.path.join(directory1, file)
            path2 = os.path.join(directory2, file_map[file_base])

            # Load the audio files
            audio1, _ = librosa.load(path1, sr=sampling_rate)
            audio2, _ = librosa.load(path2, sr=sampling_rate)

            # Compute STFT for both signals
            X = librosa.stft(audio1, n_fft=320, hop_length=160)
            Y = librosa.stft(audio2, n_fft=320, hop_length=160)

            # Compute the magnitudes
            magnitude_X = np.abs(X)
            magnitude_Y = np.abs(Y)

            # Spectral Convergence
            SC = np.linalg.norm(magnitude_X - magnitude_Y) / np.linalg.norm(magnitude_X)
            spectral_convergences.append(SC)

            # Log Spectral Distance
            LSD = np.sqrt(np.mean((np.log1p(magnitude_X) - np.log1p(magnitude_Y))**2))
            log_spectral_distances.append(LSD)

    return np.mean(spectral_convergences), np.mean(log_spectral_distances)

def calculate_for_subcategories(parent_directory1, parent_directory2, sampling_rate=16000):
    subcategories = [d for d in os.listdir(parent_directory1) if os.path.isdir(os.path.join(parent_directory1, d))]
    with open('spectral_distances.txt', 'w') as file:
        for subcat in subcategories:
            dir1 = os.path.join(parent_directory1, subcat)
            dir2 = os.path.join(parent_directory2, subcat)
            SC, LSD = calculate_spectral(dir1, dir2, sampling_rate)
            file.write(f'Subcategory: {subcat}, Spectral Convergence: {SC}, Log Spectral Distance: {LSD}\n')

# Example usage
parent_directory1 = '/home/rein/Documents/GAN-conv1d/audio'
parent_directory2 = '/home/rein/Downloads/development-dataset'
calculate_for_subcategories(parent_directory1, parent_directory2)