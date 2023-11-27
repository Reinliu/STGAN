# STGAN
STGAN: Spectral-temporal sound effects generation with GAN

## Description:
This work is done for 

## Requirements:

## Preprocess:
Configure appropriate parameters for the preprocessing and training using config.json.
You should ensure that your dataset folder contains subcategories, because our model integrates class labels as conditioning vectors.
Run 'python preprocess.py' and it will save the preprocessed features and data into a folder called 'preprocessed'

## Train:
Run 'python train.py' to train your model. 
Adjust the save_freq to save your model how many iterations you would like to save.
This will save the state-dict so you could later load them with torch.

## Generate:
Run the 'python generate.py' command to generate a number of audio samples by extracting loudness from an audio folder.
This will use the extracted loudness from an audio folder as conditioning input for audio generation. Notice the latent vector is sampled randomly. You could adjust this to generate different outputs.

## Demo:
