# 🧠 Kidney Stone Detection & Prediction System

## 📌 Overview

This project presents a machine learning-based system for the detection and prediction of kidney stones using medical data analysis. The system processes patient-related attributes and predicts the likelihood of kidney stone presence, helping in early diagnosis and decision-making.

The project was developed as a final-year academic submission and demonstrates the application of data science techniques in healthcare.

[Dataset link](https://www.kaggle.com/datasets/orvile/axial-ct-imaging-dataset-kidney-stone-detection)

## 🎯 Objectives

* Build a predictive model for kidney stone detection
* Analyze medical parameters influencing kidney stone formation
* Improve diagnostic accuracy using machine learning
* Provide an easy-to-use system for prediction



## ⚠️ Problem Statement

Traditional kidney stone diagnosis relies heavily on imaging techniques and expert interpretation, which can be:

* Time-consuming
* Costly
* Dependent on availability of specialists

This project aims to develop an automated prediction system using patient data to assist in early detection.



## 💡 Approach

The project follows a structured pipeline:

1. Data Collection
    * Dataset containing medical attributes related to kidney stone diagnosis
2. Data Preprocessing
    * Handling missing values
    * Feature selection
    * Data normalization
3. Exploratory Data Analysis (EDA)
    * Understanding feature relationships
    * Visualizing patterns
4. Model Building
    * Applied machine learning algorithms such as:
        * Logistic Regression
        * Decision Tree / Random Forest (modify if needed)
5. Model Evaluation
    * Accuracy score
    * Confusion matrix
    * Performance comparison


## 🛠️ Tech Stack

* Language: Python
* Libraries:
    * NumPy
    * Pandas
    * Matplotlib / Seaborn
    * Scikit-learn
* Environment: Jupyter Notebook



## 📂 Project Structure
```
├── Kidney_Stone_Project.ipynb   # Main notebook
├── dataset.csv                 # Dataset used
├── models/                     # Saved models (if any)
├── outputs/                    # Graphs and results
├── README.md                   # Documentation
```

⚙️ Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/kidney-stone-project.git
   ```

2. Navigate to the folder:
   ```
   cd kidney-stone-project
   ```      

3. Install required libraries:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn streamlit
   ```
4. Run streamlit app
   ```
   streamlit run app.py
   ```

## 📊 Results

* The model successfully predicts the presence of kidney stones based on input features
* Achieved good accuracy on test data (97%)
* Visualizations help in understanding key contributing factors


<img width="324" height="364" alt="image" src="https://github.com/user-attachments/assets/c3d34dfc-3573-441c-a2ca-b0c8e6e41927" />


        
