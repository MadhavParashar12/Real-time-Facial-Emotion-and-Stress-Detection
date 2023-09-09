# Real-time-Facial-Emotion-and-Stress-Detection
### Introduction
The Real-time Facial Emotion and Stress Detection project is an integrated system that leverages the capabilities of Computer Vision and Natural Language Processing (NLP) to recognize and understand human emotions and stress levels. This project employs Convolutional Neural Networks (CNN) for facial emotion detection using live video feeds through openCV and utilizes NLP techniques, including Naive Bayes, to predict stress levels from text input.

### Project Goals
1. Real-time Facial Emotion Detection: The primary goal is to build a real-time facial emotion recognition system using CNNs. The system will process live video feeds to detect and predict the emotional state of the user.

2. Stress Detection from Text: Another critical objective is to predict stress levels from user-generated text input. We will employ NLP techniques, including Naive Bayes, to analyze and classify the stress levels based on the provided text.

### Methodology
1. Facial Emotion Detection with CNN: The goal is to categorise each face into one of seven categories based on the emotion shown in the facial expression. We used the following emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
We will create a CNN-based model that is pre-trained on a large dataset of facial expressions. This model will be fine-tuned to recognize emotions in real-time video feeds captured through openCV.
Real-time Video Processing: The openCV library will be utilized to capture video streams from the user's webcam. The CNN model will process these streams frame by frame to predict emotions.

2. Stress Detection with NLP: NLP blends statistical, machine learning, and deep learning models with computational linguistics—rule-based modelling of human language. With the help of these technologies, computers are now able to process human language in the form of text or voice data and fully "understand" what is being said or written, including the speaker's or writer's intentions and sentiments.
The Dataset of this ML model has 1000 tweets and utilizes two columns: ‘label’ and ‘text’.   The texts can be labelled either 0 which means “No Stress” or 1 which means “Stress”.
The purpose of data pre-processing is to carry out data cleaning, integration, reduction, and transformation.
For stress detection, we will implement NLP techniques, such as tokenization and feature extraction. A Naive Bayes classifier will be trained on a labelled dataset to predict stress levels based on text input.

### Deliverables
1. A robust CNN model for real-time facial emotion detection.

2. A trained Naive Bayes classifier for stress prediction from text input.

### Conclusion
The Real-time Facial Emotion and Stress Detection project combines Computer Vision and NLP techniques to provide users with a holistic understanding of their emotional and stress states. By leveraging CNNs and NLP with Naive Bayes, this project offers real-time insights into human emotions and stress levels, promoting mental well-being and facilitating user self-awareness. Making it a valuable tool for both individuals and professionals in fields related to mental health and well-being assessment.
