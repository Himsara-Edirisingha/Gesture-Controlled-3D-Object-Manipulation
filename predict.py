import torch
from fastai.vision.all import *
from pathlib import Path
import matplotlib.pyplot as plt

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learn = load_learner('model\\resnet-50_2024-10-30_08.06.14.pkl', cpu= device == torch.device('cpu'))

test_images_path = Path('test images') 
test_images = list(test_images_path.glob('*.png')) + list(test_images_path.glob('*.jpg'))

if len(test_images) == 0:
    print("No test images found in the specified folder.")
else:
    for image_path in test_images:
        img = PILImage.create(image_path)
        pred, pred_idx, probs = learn.predict(img)
        print(f"Image: {image_path.name}, Prediction: {pred}, Probability: {probs[pred_idx]:.4f}")
        img.show()
        plt.show()