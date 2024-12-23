# oral_disease_detection_model
Overview
The Oral Disease Classification Model is a machine learning-based system designed to classify oral diseases from images. The model leverages deep learning techniques using Convolutional Neural Networks (CNNs) to identify specific oral conditions, such as dental caries and gingivitis, with high accuracy.

Features
Classification of oral diseases into categories: Healthy, Dental Caries, and Gingivitis.
Robust image preprocessing pipeline to handle variations in image quality, orientation, and lighting.
Transfer learning for improved performance on smaller datasets.
Interactive visualizations of training metrics (accuracy and loss).
Easy-to-extend architecture for adding new disease categories.
Technologies Used
Programming Language: Python
Frameworks & Libraries:
TensorFlow/Keras
OpenCV
NumPy
Matplotlib
Seaborn
Dataset
The dataset used for training and testing is structured as follows:

TRAIN Folder: Contains images organized into subfolders (Caries and Gingivitis).
TEST Folder: Contains images organized similarly for evaluation.
Model Architecture
The model is built using a Convolutional Neural Network (CNN) with the following structure:

Input Layer: Processes 128x128 RGB images.
Convolutional Layers: Extract features using filters of increasing depth.
MaxPooling Layers: Reduce spatial dimensions and avoid overfitting.
Dense Layers: Combine extracted features for classification.
Output Layer: Predicts the disease category using a softmax activation.
Installation
Prerequisites
Ensure that Python 3.8+ is installed on your system. Install the required dependencies:

bash
Copy code
pip install tensorflow opencv-python matplotlib seaborn
Clone the Repository
bash
Copy code
git clone https://github.com/piyushbirla10/oral_detection_model.git
cd oral-disease-classification
Usage
Training the Model
Place your dataset in the OA directory with subfolders TRAIN and TEST.
Run the training script:
bash
Copy code
python train_model.py
View training metrics (accuracy and loss) plotted during the process.
Testing the Model
Evaluate the model on the test dataset:
bash
Copy code
python evaluate_model.py
Results (accuracy, precision, recall, F1-score) and a confusion matrix will be displayed.
Making Predictions
To classify a new image:

bash
Copy code
python predict_image.py --image path/to/image.jpg
Results
Accuracy: 92.07% on the test dataset.
Confusion Matrix: Displays misclassification rates.
Classification Report: Provides precision, recall, and F1-score for each class.
Future Improvements
Expand the dataset for better generalization.
Implement advanced architectures like ResNet or EfficientNet.
Include additional disease categories for a more comprehensive classification.
Contributing
Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and create a pull request.

License
This project is licensed under the MIT License.

Contact
For queries or suggestions, feel free to contact:

Name: Piyush Birla
Email: piyushbirla2001@gmail.com
GitHub
