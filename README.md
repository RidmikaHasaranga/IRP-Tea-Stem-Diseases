# Tea Stem Disease Detection Using Machine Learning and Deep Learning

This project presents a comparative study of machine learning (ML) and deep learning (DL) approaches for the identification of tea stem diseases in Sri Lankan tea plantations.

## 📚 Project Overview
Tea stem diseases such as Stem Canker, Pink Wax Disease, and Live Wood Termite infestation pose significant threats to the tea industry. Traditional manual inspections are time-consuming and subjective. This project implements various ML and DL models to automate the detection process and improve early-stage diagnosis.

## 📂 Dataset
- Custom-built dataset with approximately **2000 original images**, expanded through **data augmentation**.
- Categories: `Healthy`, `Stem Canker`, `Pink Wax Disease`, `Live Wood Termite`
- Images captured under real field conditions in Sri Lanka.

## 🛠️ Models Implemented
- **Machine Learning Models**: Random Forest, SVM, KNN, Decision Tree (with GLCM and LBP feature extraction)
- **Deep Learning Models**: Custom CNN, ResNet50 (Transfer Learning), EfficientNetB0
- **Hybrid and Attention-Based Models**: ResNet Hybrid, CNN + SE Blocks, EfficientNet + SE Blocks
- **Grayscale-Based Models**: Training on grayscale Canny edge detected images for structural feature learning.

## 🧠 Key Techniques
- Data Augmentation (Rotation, Flipping, Zooming, Elastic Deformation)
- Transfer Learning
- Attention Mechanisms (Squeeze-and-Excitation Blocks)
- Early Stopping and Dropout Regularization

## 📊 Results
- **Best Model**: EfficientNetB0 trained on Grayscale images
- **Test Accuracy**: **88.62%**

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🖼️ Visualizations
- Confusion Matrices for Best Models
- Training vs. Validation Accuracy and Loss Graphs

## 🚀 Future Work
- Expand dataset to include more environmental variations.
- Deploy lightweight models for mobile or edge devices.
- Real-world field validation and pilot testing.

## 📜 Thesis
This project is based on the final year thesis titled  
**"Deep Learning and Machine Learning Approaches for Tea Stem Disease Identification: A Comparative Study Focused on Sri Lanka"**.

## 🧑‍💻 Author
- **Name**: Ridmika Hasaranga
- **Institution**: Informatics Institute of Technology (IIT), affiliated with Robert Gordon University (RGU)

## 📄 License
This project is for academic and research purposes only. Please cite appropriately if using any part of the dataset or code.
