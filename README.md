# Multi-Modal Video Ad Analysis with Text and Visual Integration :movie_camera:

## Overview :memo:

This project involves the analysis of video advertisements, leveraging their textual descriptions, transcriptions, and visual content to answer 21 binary (yes/no) questions. The primary objective is to develop a classifier that accurately predicts these answers, with the goal of maximizing performance metrics such as agreement percentage, F1 score, precision, and recall.

## Key Links :link:

- [Dataset - Video Advertisements](https://drive.google.com/file/d/1BJVwi50dBI6RoJKxWtqW0tU5O1FxxRsS/view?usp=share_link)
- [Dataset - Textual Data](https://drive.google.com/file/d/1TwOVtvxwpJD6toYh7bC2KKkJZDYP4Xd4/view?usp=share_link)
- [Ground Truth Data](https://docs.google.com/spreadsheets/d/1sXqrdNDuSuvF6MJw_MGMGFakcYkW-0XK/edit?usp=share_link)

## Abstract :notebook_with_decorative_cover:

The project utilizes multi-modal data from video advertisements to develop a classifier capable of answering predefined binary questions. The classifier integrates both textual and visual features extracted from the ads to maximize accuracy and consistency with human-coded ground truth data. The project also includes an evaluation of the classifier's performance using metrics such as precision, recall, F1 score, and agreement percentage.

## Project Goals :dart:

1. **Data Integration:** Combine textual and visual data from video ads to create a comprehensive feature set.
2. **Classifier Development:** Develop and train a machine learning classifier, using a multi-modal approach to answer 21 binary questions.
3. **Performance Evaluation:** Assess the classifier's performance using agreement percentage, F1 score, precision, and recall against the ground truth data.
4. **Result Documentation:** Document the entire process and results, providing insights into classifier performance and areas for improvement.

## Technologies Used :computer:

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Transformers](https://img.shields.io/badge/Transformers-007ACC?style=for-the-badge&logo=transformers&logoColor=white)](https://huggingface.co/transformers/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)](https://colab.research.google.com/)

## Methodology :gear:

1. **Data Preprocessing:**
   - Preprocessed textual data (ad descriptions, transcriptions) using NLP techniques.
   - Extracted visual features from video frames using a pre-trained VGG16 model.
   - Reduced dimensionality of features using PCA.

2. **Feature Integration:**
   - Combined textual and visual features to form a unified feature set for each video ad.

3. **Model Development:**
   - Trained XGBoost classifiers for each of the 21 questions.
   - Used class weights to handle class imbalance.
   - Optimized models using GridSearchCV.

4. **Performance Evaluation:**
   - Evaluated the model using precision, recall, F1 score, and agreement percentage.
   - Documented insights and performance variations across different questions.

## Results :chart_with_upwards_trend:

- **Average Precision:** 0.45
- **Average Recall:** 0.66
- **Average F1 Score:** 0.52
- **Average Agreement Percentage:** 63.17%

## Bonus Analysis :mag:

- **Inconsistencies in Human-Coded Data:** Analysis of human coder responses revealed inconsistencies, particularly in subjective questions.
- **Classifier Performance Issues:** Identified challenges in processing complex visuals and ambiguous content, affecting classifier accuracy.

## How to Run Locally :house:

1. **Clone the repository:**
   ```bash
   git clone <repository-link>

   
## Project Folder Structure :file_folder:
📦 Video_Ads_Analysis
├── data
│ ├── videos
│ ├── text
│ └── ground_truth
├── notebooks
│ └── Multi_Modal_Video_Ad_Analysis_with_Text_and_Visual_Integration.ipynb
├── src
│ ├── preprocessing.py
│ ├── feature_extraction.py
│ ├── model_training.py
│ └── evaluation.py
├── results
│ └── predictions.csv
└── README.md

