from frechet_audio_distance import FrechetAudioDistance
import os

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)
# # to use `PANN`
# frechet = FrechetAudioDistance(
#     model_name="pann",
#     sample_rate=16000,
#     use_pca=False, 
#     use_activation=False,
#     verbose=False
# )
# to use `CLAP`
# frechet = FrechetAudioDistance(
#     model_name="clap",
#     sample_rate=48000,
#     submodel_name="630k-audioset",  # for CLAP only
#     verbose=False,
#     enable_fusion=False,            # for CLAP only
# )

real_dataset_base = '/home/rein/Downloads/development-dataset'
generated_dataset_base = '/home/rein/Documents/GAN-conv1d/audio'


# List subcategories
subcategories = [d for d in os.listdir(real_dataset_base) if os.path.isdir(os.path.join(real_dataset_base, d))]

fad_scores = {}

for category in subcategories:
    real_path = os.path.join(real_dataset_base, category)
    generated_path = os.path.join(generated_dataset_base, category)
    
    # Compute FAD score
    fad_score = frechet.score(real_path, generated_path, dtype="float32")
    fad_scores[category] = fad_score
    print(f"Category {category}: FAD score = {fad_score}")


with open('fad_scores.txt', 'w') as file:
    for category, score in fad_scores.items():
        file.write(f"{category}: {score}\n")
