# ConvNeXt for Image Classification

This project utilizes a ConvNeXt model to classify images from a custom dataset. The ConvNeXt model is fine-tuned using transfer learning to recognize six distinct classes of image defects. This README explains the steps for preparing the dataset, training the model, and evaluating its performance.

## Overview

The ConvNeXt is a deep learning model based on a modernized ResNet architecture, which has shown excellent performance for various image classification tasks. This project involves fine-tuning a pre-trained ConvNeXt model to classify image defects into six categories. The custom dataset is split into training and test sets, with data augmentation techniques applied to enhance generalization.

### Key Features:

- **Data Augmentation**: Includes techniques such as random rotations, flips, and affine transformations to make the model more robust.
- **Transfer Learning**: Fine-tuning a pre-trained ConvNeXt model to classify specific types of defects: Crazing, Inclusion, Patches, Pitted, Rolled, and Scratches.
- **Comprehensive Evaluation**: Post-training evaluation includes accuracy, sensitivity, specificity, AUC scores, and error rate.
- **Prediction Visualization**: Visual representation of predictions made on test data to understand model performance better.

## Dataset

The dataset contains images categorized into six classes of defects:

- **Crazing**
- **Inclusion**
- **Patches**
- **Pitted**
- **Rolled**
- **Scratches**

To use this project, you need to create a folder named `custom_dateset` in the main directory and add the following subfolders: `train/` and `test/`, each containing subfolders named after the defect classes (`Crazing`, `Inclusion`, etc.) with corresponding images. The dataset structure should be as follows:

```
custom_dateset/
  ├── train/
  │     ├── Crazing/
  │     ├── Inclusion/
  │     ├── Patches/
  │     ├── Pitted/
  │     ├── Rolled/
  │     └── Scratches/
  └── test/
        ├── Crazing/
        ├── Inclusion/
        ├── Patches/
        ├── Pitted/
        ├── Rolled/
        └── Scratches/
```

Since GitHub doesn't support uploading folders directly, you'll need to create this folder structure locally and populate it with your dataset images before running the project.

The dataset must be uploaded to your Google Drive if you use Google Colab, and paths must be set accordingly.

## Training

### Model Architecture

- **Model**: ConvNeXt Base
- **Pre-trained Weights**: The model starts with pre-trained weights from ImageNet.
- **Modified Classifier Head**: The classifier head was modified to classify the six defect types.
- **Training Configuration**:
  - **Optimizer**: AdamW Optimizer with a learning rate of `1e-4` and weight decay of `1e-2`.
  - **Loss Function**: Cross Entropy Loss.
  - **Batch Size**: `16`
  - **Epochs**: `50`

### Data Augmentation Techniques

The following data augmentation techniques were applied to the training set to improve model robustness:

- **Random Rotation**: Rotates images randomly between `-45` to `45` degrees.
- **Horizontal and Vertical Flips**: Flips images randomly to augment the dataset.
- **Affine Transformations**: Includes shear and translation to generalize the model further.

## Results

The ConvNeXt model, after training on the custom dataset, achieved the following metrics:

- **Training and Validation Accuracy**: Steadily improved across epochs, with a final validation accuracy of over **99%**.
- **Confusion Matrix**: Displayed near-perfect classification for each defect class, with minimal misclassifications.
- **Evaluation Metrics**:
  - **Accuracy (AC %)**: **99.72%**
  - **Sensitivity (SE %)**: **99.72%**
  - **Specificity (SP %)**: **99.73%**
  - **Error Rate (ER %)**: **0.28%**
  - **AUC (Average)**: **1.00**

These results demonstrate the model's effectiveness in correctly identifying different defect classes with high confidence and minimal error.

## Usage Instructions

### Prerequisites

- **Python 3.7+**
- **PyTorch** and **Torchvision**
- **timm** library for ConvNeXt implementation.
- **Google Colab** or a local GPU-enabled environment is recommended for training.

### Upload the Dataset

- Create a `custom_dateset` folder in the root directory.
- Add your images following the folder structure mentioned above.

### Train the Model

If using Google Colab, ensure the dataset is properly mounted from Google Drive and run the notebook (`ConvNeXtModel.ipynb`) to start training the model.

## Project Structure

- `ConvNeXtModel.ipynb`: Main script to train the ConvNeXt model on the custom dataset.
- `custom_dateset/`: Folder containing the train and test images.

## Results Summary

The ConvNeXt model has shown excellent results, with a high validation accuracy of **99.72%** and near-perfect AUC of **1.00**, indicating outstanding performance on the custom dataset. These results will serve as a benchmark for comparison against other models, such as Vision Transformer (ViT) and FasterViT, which will be trained and evaluated similarly.

## Future Work

- **Compare with Other Models**: Further experiments will be conducted with other models like FasterViT for a comprehensive analysis.
- **Expand Dataset**: Adding more diverse images will improve model robustness.
- **Hyperparameter Tuning**: Experimenting with different hyperparameters to further optimize model performance.

## Acknowledgments

- This project utilizes the ConvNeXt implementation from the timm library.
- Thanks to the authors of ConvNeXt for their excellent work in advancing deep learning for image classification.

Feel free to fork this project, contribute, and share your insights!

