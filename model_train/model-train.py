import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from fastai.vision.all import *
from torchvision.models import resnet50
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns
from datetime import datetime

if __name__ == '__main__':
    # Set environment variables for CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Helps with debugging CUDA errors (from you tube)

    # Set random seeds for reproducibility
    torch.manual_seed(2)
    np.random.seed(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataset directory setup
    dataset_dir = Path('new-CNR-dataset')

    # Model configuration
    arch_str = "resnet-50"
    default_size = (1080, 1920)
    cnn_model = resnet50
    img_size = [int(x / 2) for x in default_size]
    epochs = 40
    bs = 8

    # Data augmentation transformations
    tfms = aug_transforms(do_flip=True, flip_vert=False, max_rotate=25., max_zoom=1.5, max_lighting=0.5, max_warp=0.1, p_affine=0.75, p_lighting=0.75)

    def create_dataset(bs, image_size=default_size, tfms=None):
        # Define transformations if not provided
        if tfms is None:
            tfms = aug_transforms(do_flip=True, flip_vert=False, max_rotate=25., max_zoom=1.5, max_lighting=0.5, max_warp=0.1)

        # Create DataLoaders
        data = ImageDataLoaders.from_folder(
            dataset_dir,
            valid_pct=0.3,
            item_tfms=Resize(image_size),
            batch_tfms=[*tfms, Normalize.from_stats(*imagenet_stats)],  # Adding normalization here
            bs=bs,
            num_workers=0  # Set num_workers to 0 to avoid multiprocessing issues on Windows
        )
        data.show_batch(max_n=9, figsize=(7, 6))
        #plt.show()  #  plot
        return data

    # Create and display dataset
    data = create_dataset(bs, img_size)

    # Model save path setup
    currtime = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    savepath = Path('models')
    savepath.mkdir(parents=True, exist_ok=True)
    model_name = arch_str
    print(f'Models will be saved at: {savepath} with name: {model_name}')

    # Create the learner with CSVLogger to save training metrics to a CSV file
    learn = vision_learner(data, cnn_model, metrics=accuracy, cbs=CSVLogger(fname="training_report.csv"))
    print('Learner ready.')

    # Train
    learn.fit_one_cycle(epochs, 1e-3)

    # Save the metrics logged by CSVLogger to disk
    # learn.csv_logger.to_csv("training_metrics.csv")

    # Remove the CSVLogger callback before exporting
    #learn.remove_cb(CSVLogger)

    # Save 
    model_save_name = f"{model_name}_{currtime}.pth"
    learn.export(savepath / f"{model_name}_{currtime}.pkl")
    print(f"Model saved as {model_name}_{currtime}.pkl at {savepath}")

    # Evaluate 
    def evaluate_model(learner):
        #predictions and targets
        preds, targs = learner.get_preds()
        acc = accuracy(preds, targs)
        print(f"Validation Accuracy: {acc.item():.4f}")

        #confusion matrix
        cm = confusion_matrix(targs, preds.argmax(dim=1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=learner.dls.vocab)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.show()

    evaluate_model(learn)

    # Predict an image
    def predict_image(image_path):
        img = PILImage.create(image_path)
        pred, pred_idx, probs = learn.predict(img)
        print(f"Prediction: {pred}, Probability: {probs[pred_idx]:.4f}")
        img.show()
        plt.show()

    # a prediction
    test_image_path = 'webcam-captures\\2024-10-30 192249.png'
    predict_image(test_image_path)

    #report
    def generate_report(learner):
        #training and validation loss graph
        learner.recorder.plot_loss()
        plt.title('Training and Validation Loss')
        plt.show()

        #predictions and true labels
        preds, targs = learner.get_preds()

        # Accuracy
        acc = accuracy(preds, targs)
        print(f"Validation Accuracy: {acc.item():.4f}")

        # Confusion Matrix
        cm = confusion_matrix(targs, preds.argmax(dim=1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=learner.dls.vocab)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.show()

        # Classification Report
        report = classification_report(targs, preds.argmax(dim=1), target_names=learner.dls.vocab)
        print("Classification Report:\n", report)

        report_save_path = Path("model_reports")
        report_save_path.mkdir(parents=True, exist_ok=True)
        report_file = report_save_path / f"training_report_{currtime}.txt"
        with open(report_file, "w") as f:
            f.write(f"Validation Accuracy: {acc.item():.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        print(f"Training report saved to {report_file}")

    generate_report(learn)
