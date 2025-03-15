Brain Tumor Detection Using CNN
Project Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow to detect and classify brain tumors from MRI scans. The model achieves 96% accuracy by leveraging advanced image preprocessing, data augmentation, and transfer learning techniques.

Key Features
Deep Learning for Medical Imaging: Designed a CNN architecture to accurately classify brain tumors.
Optimized Model Performance: Fine-tuned hyperparameters, experimented with activation functions, dropout, and batch normalization.
Advanced Image Preprocessing: Applied data augmentation, contrast enhancement, and noise reduction to improve model generalization.
High Classification Accuracy (96%): Used transfer learning (EfficientNet, ResNet) to enhance model performance.
End-to-End Pipeline: Includes data preprocessing, model training, evaluation, and visualization using TensorFlow and Keras.
Dataset
The model is trained on publicly available brain MRI scan datasets, pre-labeled with tumor presence/absence.
Images are resized, normalized, and augmented to improve training efficiency and robustness.
Model Architecture
Input Layer: MRI scan images (grayscale/RGB)
CNN Layers: Multiple convolutional layers with ReLU activation and max pooling
Fully Connected Layers: Dense layers with softmax activation for multi-class classification
Optimization Techniques: Adam optimizer, categorical cross-entropy loss function
Implementation Details
1. Data Preprocessing
Resized and normalized MRI scan images
Applied data augmentation (rotation, flipping, zooming)
Enhanced contrast and reduced noise using image processing techniques
2. Model Training
Implemented a CNN from scratch using TensorFlow and Keras
Trained using Adam optimizer and learning rate scheduling
Used dropout and batch normalization to prevent overfitting
3. Performance Evaluation
Achieved 96% accuracy on test data
Evaluated using precision, recall, F1-score, and confusion matrix
Visualized Grad-CAM heatmaps to interpret CNN predictions
Results
Training Accuracy: ~98%
Validation Accuracy: ~96%
Precision and Recall: High across all classes
Confusion Matrix: Minimal false positives and false negatives
Future Improvements
Implement attention mechanisms for better feature extraction
Extend to multi-class tumor classification (glioma, meningioma, pituitary)
Deploy the model as a web-based application for real-world usage
Technologies Used
Python, TensorFlow, Keras (Deep Learning)
OpenCV, NumPy, Pandas (Data Processing)
Matplotlib, Seaborn (Visualization)
