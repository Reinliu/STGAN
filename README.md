# STGAN: Spectral-temporal sound effects generation with RNN-GAN
This repository provides implementations for a lecture presentation and a conference proceedings at the Acoustics 2023 Sydney conference. 

We use RNN for the sound generator. The sound generation is guided and conditioned by a loudness envelope and a class label. We use a vanilla [WGAN-GP](https://github.com/gcucurull/cond-wgan-gp/) discriminator to compute the adversarial loss on the generated Mel Spectrograms. 
For the generator loss, we further include a MSE reconstruction loss on the loudness difference between the generated and original Mel Spectrogram.
The generated Mel spectrogram can be converted back to audio using a pretrained HIFIGAN vocoder from [AudioLDM](https://github.com/haoheliu/AudioLDM/).

## Model architecture
### Training: 
![image](https://github.com/Reinliu/STGAN/assets/50271800/b8e3594b-14e5-4a6e-b952-0cafbe35a9a5)



### Inference:
![image](https://github.com/Reinliu/STGAN/assets/50271800/6bec786e-b4e0-4f3c-9e37-c69fc739fdca)




## Requirements:
~~~
pip install -r requirements.txt
~~~

## Preprocess:
Configure appropriate parameters for the preprocessing and training using config.json.
You should ensure that your dataset folder contains subcategories, because our model integrates class labels as conditioning vectors.
Run 'python preprocess.py' and it will save the preprocessed features and data into a folder called 'preprocessed'

## Train:
Run 
~~~
python train.py
~~~
to train your model. 
Adjust the save_freq to save your model how many iterations you would like to save.
This will save the state-dict so you could later load them with torch.

## Generate:
Run 
~~~
python generate.py
~~~
command to generate a number of audio samples by extracting loudness from an audio folder.
This will use the extracted loudness from an audio folder as conditioning input for audio generation. Notice the latent vector is sampled randomly. You could adjust this to generate different outputs.
