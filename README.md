**# CNN Image Classification with TensorFlow & Keras**


This project implements an image classification pipeline using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. It is designed to classify images from a directory-based dataset and demonstrates data loading, preprocessing, model building, and training.

## 📁 Project Structure

```
├── kaggle.json         # Kaggle API token
├── train/              # Training image directory (class subfolders inside)
├── test/               # Validation image directory (class subfolders inside)
├── cnn_model.py        # (Optional) Script containing model definition and training
└── README.md
```

## 🚀 Features

* Data loading using `image_dataset_from_directory`
* Normalization of pixel values
* CNN model with Conv2D, MaxPooling, BatchNormalization, Dropout
* Binary classification using sigmoid activation
* Performance monitoring using validation accuracy/loss

## 📦 Setup Instructions

1. **Install dependencies**

   ```bash
   pip install tensorflow
   ```

2. **Set up Kaggle API (optional)**
   If using datasets from Kaggle:

   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Prepare dataset**
   Ensure your training and validation datasets are organized as follows:

   ```
   train/
     class_0/
     class_1/
   test/
     class_0/
     class_1/
   ```

## 🧠 Model Architecture

| Layer                | Output Shape   | Parameters |
| -------------------- | -------------- | ---------- |
| Conv2D (32 filters)  | (254, 254, 32) | 896        |
| BatchNormalization   | (254, 254, 32) | 128        |
| MaxPooling2D         | (127, 127, 32) | 0          |
| Conv2D (64 filters)  | (125, 125, 64) | 18,496     |
| BatchNormalization   | (125, 125, 64) | 256        |
| MaxPooling2D         | (62, 62, 64)   | 0          |
| Conv2D (128 filters) | (60, 60, 128)  | 73,856     |
| BatchNormalization   | (60, 60, 128)  | 512        |
| MaxPooling2D         | (30, 30, 128)  | 0          |
| Flatten              | (115200,)      | 0          |
| Dense (128 units)    | (128,)         | 14,745,728 |
| Dropout (0.1)        | (128,)         | 0          |
| Dense (64 units)     | (64,)          | 8,256      |
| Dropout (0.1)        | (64,)          | 0          |
| Output (Sigmoid)     | (1,)           | 65         |

**Total Parameters**: 14.8 million
**Trainable**: 14,847,745
**Non-trainable**: 448

## 🏃‍♀️ Training Performance

* Optimizer: `Adam`
* Loss: `Binary Crossentropy`
* Epochs: 10
* Batch Size: 32
* Image Size: 256x256

### Sample Results:

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
| ----- | -------------- | ------------ | ---------- | -------- |
| 1     | 0.5526         | 0.5806       | 3.0269     | 1.0544   |
| 5     | 0.8290         | 0.7690       | 0.3777     | 0.5234   |
| 10    | 0.9638         | 0.7674       | 0.0947     | 0.8854   |

> The model shows strong performance on the training set and reasonable generalization on the validation set. Further improvements could include data augmentation and early stopping.

## 📌 Notes

* Ensure binary classification (i.e., two folders under each of `train/` and `test/`).
* Adjust `Dense(1)` to `Dense(num_classes, activation='softmax')` if you want multi-class classification.
* Validation accuracy may fluctuate due to limited dataset or overfitting; tuning dropout and layers can help.
