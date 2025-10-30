# COMP3710 Report

# Task 3 - 2D Prostate Segmentation with U-Net

## Guanhua Ma 48339261



## 1. Project Description & Problem

This project aims to solve the Task 3 (Normal Difficulty) pattern recognition problem.

This project uses the processed 2D slices from the HipMRI Study on Prostate Cancer dataset to perform automatic semantic segmentation of the prostate region. The goal is to train a 2D Improved UNet to achieve a minimum Dice Similarity Coefficient (DSC) of 0.75 on the prostate label in the test set.

## 2. Project File

These are Project Files:

- `modules.py`: Contains the definitions for the `SimpleUNet` model architecture and the `DiceLoss` function.
- `dataset.py`: Contains the `HipMRIDataset` class, responsible for loading and preprocessing the Nifti data.
- `train.py`: Contains the main training loop `train()`, the train/validation split logic.
- `predict.py`: Contains the `show_predictions()` function to load the trained model and visualize its performance on test samples.
- `utils.py`: Contains helper functions such as `calculate_dice_score()`, `show_epoch_predictions()`, and `plot_loss()`.
- `README.md`: The report document for this project.

## 3. How it Works & Algorithm

#### Algorithm Model

This section is based on the SimpleUNet model defined in modules.py. This is a classic 2D U-Net architecture , which is an encoder-decoder network. Its key feature is the use of skip connections. This concatenates feature maps from the encoder (down-sampling path) with the decoder (up-sampling path). UNet allows the model to use both semantic features and  spatial features, making it ideal for medical image segmentation.

The model's main components include:

- Encoder: A series of conv_block (Conv -> BatchNorm -> LeakyReLU -> Dropout) and MaxPool2d layers to extract features and reduce spatial dimensions.
- Decoder: Uses Upsample (Bilinear Interpolation) and _conv_block layers to reconstruct the segmentation mask.
- Output Layer: A final Conv2d layer followed by a Sigmoid activation function to output a probability map in the range [0, 1].

The Loss Function is Dice Loss (1 - Dice Coefficient), which directly optimizes DSC. It is well-suited for imbalanced classes. Therefore, it is very suitable for this task since the prostate region is much smaller than the background.

The optimizer is Adam. Adam is an efficient and commonly used gradient descent optimizer.

#### Data Preprocessing

1. Data Loading: Uses the nibabel library to load .nii.gz Nifti format images and masks.
2. Label Binarization: The HipMRI masks are multi-class. To solve Task 3, pixels with the `prostate_label_value` (set to 5 in the code) are mapped to 1, and all other pixels are mapped to 0. This creates a binary prostate vs. non-prostate mask.
3. Resizing: All images and masks are resized to a fixed (128, 128) size.
   - Images are resized using Bilinear interpolation.
   - Masks are resized using Nearest Neighbor interpolation to ensure the label values (0 and 1) are not corrupted.
4. Normalization: Z-score normalization ((image - mean) / std) is applied to the images to set their mean to 0 and standard deviation to 1.

This dataset keras_slices_data folder has already split in three subset: keras_slices_train, keras_slices_validate and keras_slices_test. 
For this structure, the parameter `subset` is used to construct training set and validation set.

`subset="train"`: loading the keras_slices_train folder.

`subset="validate"`: loading the keras_slices_validate folder

## 4. Reproducibility

This project was run on the Google Colab with T4 GPU orA100 GPU

The main dependencies are:

```
torch (PyTorch)
numpy
matplotlib
nibabel (!pip install nibabel)
tqdm
```

Due to the time queue in Rangpur is too long, so this project choose to run in the Goole Colab

The files `modules.py`, `dataset.py`, `train.py`, `predict.py`, `utils.py` and `keras_slices_data` and `Run_on_Colab.ipynb` need to be in the same folder in Google Drive.

Open the `Run_on_Colab.ipynb` and run for the whole task.

Make sure that the path is /content/drive/MyDrive/Colab-Notebooks/UNet_task3_48339261/modules.py, for example for `modules.py`. 

This is the processing code in  `Run_on_Colab.ipynb`. 

```
# Run Task 3 in Google Colab

from google.colab import drive
drive.mount('/content/drive', force_remount=True) 

!pip install nibabel -q 

import os
base_dir = "/content/drive/MyDrive/Colab-Notebooks/UNet_task3_48339261"
os.chdir(base_dir)

# load dataset and train model. save hipmri_unet_model.pth and .png 
print("Starting process train.py")
!python train.py

# save final_predictions.png
print("Starting process predict.py")
!python predict.py

print("Successful!")
print("Check the documents:")
print("hipmri_unet_model.pth")
print("training_loss_curve.png")
print("epoch_X_predictions.png")
print("final_predictions.png")
```

### 5. Results & Analysis

#### Training Loss Curve

The training loss (Dice Loss) over time is shown below in Figure 1. The loss steadily decreases from an initial average of 0.6089 and successfully converges to a final average loss of 0.1496. This indicates that the model learned effectively from the training data.

![training_loss_curve (1)](/Users/maguanhua/Downloads/training_loss_curve (1).png)

[Figure 1: Training loss curve]

#### Prediction Visualization

The figure below shows the final segmentation performance of the model on 3 random samples from the test set after training for 20 epochs.

![final_predictions](/Users/maguanhua/Downloads/final_predictions (1).png)

[Figure 2: Final predictions]

After training for 20 epochs, the model achieved a final average Dice Loss of 0.1496 on the training set, which corresponds to an average DSC of **0.8504**.

This result exceeds the target of 0.75.

The model's performance on the random test samples in Figure 2: Final predictions is excellent。It correctly identified two **True Negatives** (Original 110 and 39). These two samples with no prostate was present in the ground truth, and predicted an empty mask. Sample 110 and 39 result in perfect Dice scores of 1.000. The model only failed on Sample 315, which was a very small and challenging target, resulting in a Dice score of 0.000.

Overall, the high average DSC (0.8504) and the strong performance on True Negatives confirm that the model successfully learned to segment the prostate gland.