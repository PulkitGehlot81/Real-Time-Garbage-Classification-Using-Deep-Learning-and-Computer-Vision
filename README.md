# Real-Time-Garbage-Classification-Using-Deep-Learning-and-Computer-Vision

# Garbage Classification using Deep Learning
This project aims to classify garbage into six categories: cardboard, glass, metal, paper, plastic, and trash, using deep learning techniques. The garbage classification model is built using TensorFlow and Keras libraries, and it is trained on a dataset consisting of 2,521 images.

# Dataset
The dataset used in this project is obtained from Kaggle. It contains images of garbage from six categories and is split into 80% training and 20% validation set.

# Model Architecture
The model architecture consists of three convolutional layers, each followed by a max-pooling layer. The output of the third max-pooling layer is then flattened and fed into two dense layers with ReLU activation functions, where the final dense layer has six neurons (one for each category) with a softmax activation function.

# Data Augmentation
Data augmentation techniques such as horizontal flip, vertical flip, rotation, zoom, width and height shift, and shear are applied to increase the model's robustness.

# Evaluation
The model is evaluated on the validation set, and the metrics used are categorical cross-entropy loss and accuracy. The accuracy achieved on the validation set is 84%.

# Results
The model achieves a reasonably good accuracy in classifying the garbage into six categories. The accuracy and loss curves show that the model is not overfitting and is generalizing well.

# Conclusion
Garbage classification using deep learning can be an effective way of reducing the negative impact of waste on the environment. By automatically classifying garbage into different categories, it can help in proper disposal and recycling.
