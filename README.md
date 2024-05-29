# Painting Vs Digital Image Classification

This project consists of two main components: a machine learning model for predicting whether an image is a painting or a digital image, and a web interface to interact with the model.


## Overview

The project includes:
1. **Machine Learning Model**: A custom EfficientNet B4 model combined with texture features extracted using HOG (Histogram of Oriented Gradients).
2. **Web Interface**: A Flask-based web application to upload images and predict if they are paintings or digital images.

## Directory Structure

PaintingVsDigitalImageClassification/
- static/
   - uploads/  
- templates/
   - upload.html
   - result.html
- app.py  
- model/
   - B4Model_saved.pth  
- datasets/
   - train/ 
   - validation/ 
   - test/
- train_model.py   
- requirements.txt 

## Setup Instructions in Visual Studio Code

### Clone the Repository

git clone https://github.com/Nandini0387/Painting-Vs-Digital-Image-Classification.git


### Set Up Virtual Environment

cd painting-vs-digital-image-prediction


### Create and Activate Virtual Environment:

python -m venv venv

source ../venv/bin/activate  # On Windows, use ..\venv\Scripts\activate

### Install Dependencies

pip install -r requirements.txt

### Dataset

Download the dataset from this link to [dataset](https://drive.google.com/drive/folders/1plINLSLzs73p16Vp3dUex1-UYQQn0ZeK?usp=drive_link).

## Training the Model
Navigate to the Project Directory and run:

cd painting-vs-digital-image-prediction 

python train_model.py

## Training Output
Training losses and accuracies will be plotted.
A trained model will be saved in the model/ directory.

## Steps to Run the Web Interface in VS code
1. Activate Virtual Environment:

  source ../venv/bin/activate  # On Windows, use ..\venv\Scripts\activate

2. Install Dependencies:

  pip install -r requirements.txt

3. Run the Flask App:

  python app.py

4. Access the Web Interface:

  Open a web browser.
  Go to http://localhost:5000 to access the web interface.
