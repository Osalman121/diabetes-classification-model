
Diabetes Prediction Model

This repository contains code for predicting diabetes using machine learning algorithms.

Introduction:

Diabetes is a chronic disease that affects millions of people worldwide. Early detection and management of diabetes are crucial for preventing complications. Machine learning techniques can assist in predicting diabetes based on various factors.

Data

The dataset used for training the model is provided in the file Training.csv. It contains the following columns:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1) indicating whether the person has diabetes or not

Setup
Make sure you have Python installed, along with the following libraries:

numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
You can install these libraries using pip:

pip install numpy pandas matplotlib seaborn scipy scikit-learn

Instructions:

Clone this repository to your local machine.
Ensure that the dataset Training.csv is placed in the same directory as the code file.
Run the code in a Python environment. It will load the dataset, perform data analysis, and train various machine learning models.

The trained models include:

Support Vector Machine (SVM)
K Nearest Neighbors (KNN)
Decision Tree
Random Forest
Logistic Regression
The accuracy of each model is displayed, and a visualization showing the accuracy variation with different parameters (for KNN and Random Forest) is provided.

Results:

Accuracy using Support Vector Machine (SVM): 72.7%
Accuracy using K Nearest Neighbors (KNN): 97.15%
Accuracy using Decision Tree: 100%
Accuracy using Random Forest: 100%
Accuracy using Logistic Regression: 72.9%

Conclusion:

Based on the evaluation, the Random Forest model achieved the highest accuracy among the tested models. However, further fine-tuning and optimization may be required for better performance. This model can be used for predicting diabetes in patients based on their health parameters.

Note:

This code is for educational purposes and should not be used as a substitute for professional medical advice.





