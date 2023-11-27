import os
import subprocess

real_dataset_base = '/home/rein/Documents/GAN-conv1d/images/'
generated_dataset_base = '/home/rein/Downloads/dev-data-melspecs/'

subcategories = [d for d in os.listdir(real_dataset_base) if os.path.isdir(os.path.join(real_dataset_base, d))]

fid_scores = {}
for category in subcategories:
    real_path = os.path.join(real_dataset_base, category)
    generated_path = os.path.join(generated_dataset_base, category)

    command = f"python3 -m pytorch_fid {real_path} {generated_path}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        fid_score = result.stdout.strip().split('\n')[-1]
        fid_scores[category] = fid_score
        print(f"Category {category}: FID score = {fid_score}")
    else:
        print(f"Error in computing FID for {category}: {result.stderr}")

with open('fid_scores.txt', 'w') as file:
    for category, score in fid_scores.items():
        file.write(f"{category}: {score}\n")
