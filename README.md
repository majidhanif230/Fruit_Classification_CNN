# Fruits Classifier
This repository contains a fruits classification project using TensorFlow and Keras. The model is trained to classify images of four types of fruits: apples, bananas, mangoes, and oranges. The dataset includes 800 images, with 640 used for training and 160 for testing. Various techniques such as dropout, data augmentation, and regularization are applied to improve the model's accuracy. MobileNet settings are utilized in the custom model for efficient performance.
# Requirements
Python 3.x
TensorFlow
Keras
NumPy
Pandas
OpenCV
Jupyter Notebook
Installation

# Clone the repository:


git clone https://github.com/majidhanif230/Fruit_Classification_CNN.git

cd Fruit_Classification_CNN
# Create a virtual environment and activate it:


python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
# Install the required packages:


pip install -r requirements.txt
# Usage
Open the Jupyter Notebook:

jupyter notebook Fruits_Classifier.ipynb
Follow the steps in the notebook to train the model and evaluate its performance.

To classify new images, use the trained model as demonstrated in the notebook.

# Results
The model achieved high accuracy on the test set by leveraging various data augmentation and regularization techniques. Detailed results and performance metrics are provided in the notebook.
# Contributors
This project was given by **Machine Learning 1 Pvt Limited** and created by a team of members:
Majid Hanif
Usama Shafeeq (Team Lead)
Ammar Rizwan
Muhammad Shahzaib

# License
This project is free to use. See the LICENSE file for details.
