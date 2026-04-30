# Speech Emotion Recognition using CNN 

This module is part of the **Multimodal Sarcasm Detection System**, focusing on extracting emotions from speech signals using deep learning techniques.

---

## Overview

The audio model processes speech input and predicts emotional states such as:

- Joy 😊  
- Sadness 😔  
- Anger 😠  
- Fear 😨  
- Surprise 😲  
- Disgust 🤢  
- Neutral 😐  

These emotions are later mapped to sentiment and compared with text sentiment for sarcasm detection.

---

## Model Used

- **Convolutional Neural Network (CNN)**
- Input: Audio features (Mel Spectrogram + Delta + Delta-Delta)
- Output: Emotion classification

---

## Features

- Converts speech into image-like representations  
- Captures both spectral and temporal patterns  
- Works with uploaded audio and live microphone input  
- Robust emotion detection (~86.1% accuracy)  


## 📦 Download Models

Due to size limitations, models are hosted externally:

- CNN Audio Model: https://drive.google.com/drive/folders/1XLFZdJu-qQKdp-qLvkAPQfR2tg5kfVfo?usp=sharing

