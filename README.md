# Hybrid 3D-CNN + Transformer for Fight Detection

This project combines a 3D Convolutional Neural Network (3D-CNN) and a Transformer Encoder to classify videos into two categories: Fight and Non-Fight.

- **3D-CNN:** Extracts spatial and motion-related features from video clips.
- **Transformer:** Learns relationships between frame-level features.
- **Classifier:** Predicts the final video class.

Dataset: RWF-2000 (Real-world Violence Dataset)

Framework: PyTorch


CELL 1: The cell defines the key model settings. CUDA will be selected if a compatible GPU is available; otherwise, the model runs on the CPU.
CELL 2:this is feature-extraction component. It processes video clips and returns a sequence of 128-dimensional feature vectors, one for each frame.(defining 3D-CNN)