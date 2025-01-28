# 🖼️ Face Recognition with PyTorch

Welcome to my face recognition project! This repository showcases a custom Convolutional Neural Network (CNN) built using PyTorch to identify faces with an impressive accuracy of **92.5%**. From dataset preprocessing to model training and evaluation, everything you need to replicate this experiment is here. 😊

---

## 🚀 Features of the Project
- **Data Preprocessing**: 
  - Images resized to 128x128, converted to grayscale, and augmented with rotations, flips, and more.
  - Normalization for smoother training. 
- **Model Architecture**:
  - A simple yet effective CNN with dropout for regularization.
  - Fully connected layers to classify faces into 40 classes.
- **Training & Validation**:
  - Clear visualization of training and validation losses across 50 epochs.
  - Dynamic graph updates for easy debugging.
- **Testing**:
  - Achieved a test accuracy of **92.5%** on unseen data.

---

## 🧑‍💻 What Makes This Project Special?

1. **PyTorch Simplicity**: 
   - PyTorch's dynamic computation graph makes it intuitive and easy to debug.
   - Perfect for building and experimenting with custom architectures.

2. **Real-World Challenges Solved**:
   - **Data Handling**: Custom data loader for seamless preprocessing and augmentation.
   - **Transparent Debugging**: You can print tensors, view shapes, and track computations—no hidden complexity.

3. **Clean & Modular Code**:
   - Organized scripts for training, validation, and testing.
   - Simple functions that are easy to adapt to your dataset.

---

## 🛠️ Getting Started

Follow these steps to set up and run the project on your system:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Asaha9345/Face_recognition_system.git
cd face-recognition-pytorch
```

### 2️⃣ Install Dependencies
Make sure you have Python 3.8+ and the required libraries:
```bash
pip install torch torchvision matplotlib opencv-python
```

### 3️⃣ Prepare the Dataset
Place your dataset (e.g., `Face_recognition_dataset.zip`) in the project directory. Unzip the folder or let the script handle it:
```python
with zipfile.ZipFile('path/to/your_dataset.zip') as files:
    files.extractall()
```

### 4️⃣ Run the Code
Train the model:
```bash
python train.py
```

Evaluate on the test set:
```bash
python test.py
```

---

## 📊 Results

- **Training & Validation Loss**: Clear convergence over 50 epochs (check `loss_plot.png`).
- **Accuracy**: Achieved **92.5%** test accuracy after fine-tuning.
- **Loss Visualization**:

## 🧠 Lessons Learned

1. **Data Augmentation** is key to improving model generalization.
2. **PyTorch’s Dynamic Graph** is a game-changer for custom model debugging and experimentation.
3. **Model Regularization** with dropout prevents overfitting and ensures stability.

---
