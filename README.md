# 🏦 Loan Approval Prediction

End-to-end Machine Learning project to predict whether a loan application is likely to be approved or rejected based on applicant, financial, employment, credit history, and property-related information.

**Assessment-1 | Machine Learning Project**

---

## 📋 Project Overview

| **Component**                  | **Description**                                        | **Marks** |
| ------------------------------ | ------------------------------------------------------ | --------: |
| 1. Problem Identification      | Loan approval problem, objectives, success metrics     |        10 |
| 2. Dataset & Preprocessing     | Dataset cleaning, missing-value handling, encoding     |        15 |
| 3. EDA & Visualization         | Dataset analysis, distributions, feature relationships |        10 |
| 4. ML Algorithm Implementation | Multiple classification algorithms                     |        20 |
| 5. Model Evaluation            | Accuracy, Precision, Recall, F1, Confusion Matrix      |        10 |
| 6. Model Improvement           | Model selection, tuning and performance improvement    |        10 |
| 7. Application / UI            | Streamlit web application                              |        10 |
| 8. GitHub Repository           | Repository structure, README, code and files           |         5 |
| 9. Deployment                  | Streamlit Cloud deployment                             |         5 |
| 10. Presentation & Viva        | PPT and project explanation                            |         5 |
| **Total**                      |                                                        |   **100** |

---

# 🎯 1. Problem Identification

## Business Problem

Loan approval is an important process in banks and financial institutions. It requires evaluating several factors such as applicant income, coapplicant income, education, employment type, loan amount, credit history, and property area.

Manual evaluation of loan applications can be time-consuming and may lead to inconsistent decisions. A Machine Learning based system can analyze historical loan application data and predict whether a new application is likely to be approved or rejected.

## Objective

Build a binary classification Machine Learning model that predicts the loan approval status of an applicant based on demographic, employment, financial, credit, and property-related features.

## Success Metrics

The performance of the classification models is evaluated using:

* **Accuracy** – Overall percentage of correct predictions
* **Precision** – Percentage of predicted approvals/rejections that are correct
* **Recall** – Ability of the model to correctly identify the actual class
* **F1-Score** – Balance between precision and recall
* **Confusion Matrix** – Detailed classification performance

The final model is selected based on its overall classification performance.

## Stakeholders

* Banks and financial institutions
* Loan approval departments
* Credit analysts
* Financial service providers
* Customers applying for loans

---

# 📊 2. Dataset & Preprocessing

## Dataset Information

**File:** `loan_approval_dataset.csv`

**Dataset Size:** 1,005 rows × 14 columns

The dataset contains applicant information, financial information, employment details, credit history, property area, and the final loan approval status.

### Dataset Columns

* `Loan_ID`
* `Gender`
* `Married`
* `Dependents`
* `Education`
* `Self_Employed`
* `EmploymentType`
* `ApplicantIncome`
* `CoapplicantIncome`
* `LoanAmount`
* `Loan_Amount_Term`
* `Credit_History`
* `Property_Area`
* `Loan_Status`

## Features

### Demographic Features

* Gender
* Married
* Dependents
* Education

### Employment Features

* Self_Employed
* EmploymentType

### Financial Features

* ApplicantIncome
* CoapplicantIncome
* LoanAmount
* Loan_Amount_Term

### Credit Feature

* Credit_History

### Property Feature

* Property_Area

### Target

* `Loan_Status`

The target represents whether the loan application is approved or rejected.

---

## Preprocessing Steps

The following preprocessing operations were performed:

1. **Duplicate Removal**

   * The dataset initially contained 5 duplicate rows.
   * Duplicate records were removed.

2. **Missing Value Handling**

   * Missing categorical values were replaced using the mode.
   * Missing numerical values were replaced using the median.

3. **Loan ID Removal**

   * `Loan_ID` was removed because it is a unique identifier and does not provide useful predictive information.

4. **Categorical Encoding**

   * Categorical variables were converted into numerical values using encoding techniques.

5. **Feature and Target Separation**

   * Input features were separated from the target variable `Loan_Status`.

6. **Train-Test Split**

   * The dataset was divided into training and testing sets.
   * A stratified split was used to maintain class distribution.

7. **Feature Scaling**

   * Numerical features were scaled where required for algorithms such as Logistic Regression and KNN.

---

# 🔍 3. EDA & Visualization

Exploratory Data Analysis was performed to understand the characteristics of the loan dataset and identify relationships between applicant features and loan approval status.

## EDA Objectives

* Understand the distribution of loan approvals
* Analyze applicant income
* Analyze loan amount
* Examine credit history
* Study property area distribution
* Identify relationships between categorical features and loan status
* Identify important factors affecting loan approval

## Key Analysis Areas

The following visualizations can be included in the project:

* Loan Status distribution
* Gender distribution
* Education distribution
* Applicant Income distribution
* Loan Amount distribution
* Credit History distribution
* Property Area distribution
* Loan Status by Education
* Loan Status by Credit History
* Loan Status by Property Area
* Loan Status by Employment Type
* Correlation heatmap

## Expected Insights

Important factors such as **Credit History, Applicant Income, Loan Amount, Education, Employment Type, and Property Area** can influence the predicted loan approval status.

Credit history is particularly useful because it provides information about an applicant's previous credit behavior.

---

# 🤖 4. ML Algorithm Implementation

Multiple classification algorithms were implemented and compared.

| **Model**           | **Why Chosen**                                             |
| ------------------- | ---------------------------------------------------------- |
| Logistic Regression | Simple and interpretable classification baseline           |
| Decision Tree       | Captures non-linear relationships and is easy to interpret |
| Random Forest       | Ensemble model with strong performance on tabular data     |
| K-Nearest Neighbors | Distance-based classification method useful for comparison |

## Logistic Regression

Logistic Regression was used as the baseline classification model.

It estimates the probability of an applicant belonging to a particular loan-status class.

## Decision Tree

Decision Tree classifies applications using a sequence of decision rules based on input features.

## Random Forest

Random Forest combines multiple decision trees to improve prediction performance and reduce overfitting.

## K-Nearest Neighbors

KNN predicts the class of a new applicant based on the classes of nearby training samples.

---

# 📈 5. Model Evaluation

The trained models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## Model Comparison

After training the models, their performance is compared.

| **Model**           |     **Accuracy** |    **Precision** |       **Recall** |     **F1-Score** |
| ------------------- | ---------------: | ---------------: | ---------------: | ---------------: |
| Logistic Regression | Add actual value | Add actual value | Add actual value | Add actual value |
| Decision Tree       | Add actual value | Add actual value | Add actual value | Add actual value |
| Random Forest       | Add actual value | Add actual value | Add actual value | Add actual value |
| KNN                 | Add actual value | Add actual value | Add actual value | Add actual value |

> **Note:** Replace the values above with the actual results obtained from your Colab model evaluation.

## Confusion Matrix

The confusion matrix is used to identify:

* True Positive
* True Negative
* False Positive
* False Negative

It provides a detailed view of how accurately the model classifies approved and rejected loan applications.

---

# 🚀 6. Model Improvement

Model performance can be improved by selecting the best-performing algorithm and optimizing its parameters.

## Improvement Techniques

1. Compare multiple Machine Learning algorithms.
2. Select the best-performing model.
3. Tune important hyperparameters.
4. Evaluate the tuned model using the test dataset.
5. Save the final trained model using Joblib.

## Final Model

The final model is selected based on the evaluation results.

**Final Model:** `Add your best model name here`

**Test Accuracy:** `Add actual accuracy`

**Precision:** `Add actual precision`

**Recall:** `Add actual recall`

**F1-Score:** `Add actual F1-score`

The selected model is saved as:

```text
models/final_loan_approval_model.pkl
```

---

# 💻 7. Application / UI

A Streamlit web application was developed to provide an interactive interface for loan approval prediction.

## Application Features

The application allows users to enter:

### Applicant Information

* Gender
* Married status
* Dependents
* Education
* Self Employment
* Employment Type

### Financial Information

* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area

After entering the information, the user can click:

**🔮 Predict Loan Status**

The application displays:

* **✅ LOAN APPROVED**
* **❌ LOAN REJECTED**

The application also displays the processed input data for transparency.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit

## Run Locally

```bash
cd loan-approval-prediction
pip install -r requirements.txt
py -m streamlit run app/app.py
```

---

# 📁 8. GitHub Repository Structure

The project repository is organized as follows:

```text
loan-approval-prediction/
│
├── dataset/
│   ├── loan_approval_dataset.csv
│   └── processed_loan_data.csv
│
├── notebooks/
│   ├── loan_preprocessing.ipynb
│   └── loan_model_development.ipynb
│
├── models/
│   └── final_loan_approval_model.pkl
│
├── results/
│   └── model_comparison.csv
│
├── app/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Repository Contents

### `dataset/`

Contains the original and processed loan datasets.

### `notebooks/`

Contains the Google Colab/Jupyter notebooks used for preprocessing and model development.

### `models/`

Contains the trained Machine Learning model.

### `results/`

Contains model evaluation and comparison results.

### `app/`

Contains the Streamlit application.

### `requirements.txt`

Contains the Python libraries required to run the application.

---

# 🌐 9. Deployment

## Streamlit Community Cloud

The Loan Approval Prediction application was deployed using Streamlit Community Cloud.

### Deployment Steps

1. Upload the project to GitHub.
2. Create a Streamlit Community Cloud application.
3. Connect the GitHub repository.
4. Select the `main` branch.
5. Select:

```text
app/app.py
```

6. Deploy the application.
7. Test the deployed application using different applicant inputs.

## Live Application

**Streamlit App:**
`Add your deployed Streamlit URL here`

Example:

```text
https://your-loan-app.streamlit.app
```

## Local Execution

The application can also be executed locally using:

```bash
streamlit run app/app.py
```

---

# 🎤 10. Presentation & Viva

The project presentation covers the complete Machine Learning workflow.

## Presentation Topics

* Problem identification
* Business problem
* Dataset overview
* Data preprocessing
* Exploratory Data Analysis
* Machine Learning algorithms
* Model comparison
* Model evaluation
* Final model selection
* Streamlit application
* GitHub repository
* Deployment
* Results
* Future improvements



# 🛠 Tech Stack

* **Python 3**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Matplotlib**
* **Streamlit**
* **Git**
* **GitHub**
* **Streamlit Community Cloud**

---

# 📌 How to Reproduce

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd loan-approval-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

The application will open in the browser.

---

# 📊 Project Workflow

```text
Loan Dataset
     ↓
Data Cleaning
     ↓
Remove Duplicates
     ↓
Handle Missing Values
     ↓
Remove Loan_ID
     ↓
Encode Categorical Features
     ↓
Feature & Target Separation
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Save Trained Model
     ↓
Streamlit Application
     ↓
GitHub Repository
     ↓
Streamlit Cloud Deployment
     ↓
Loan Approval Prediction
```

---

# 📌 Conclusion

The Loan Approval Prediction System demonstrates an end-to-end Machine Learning workflow for predicting loan approval status. The project includes dataset preprocessing, exploratory analysis, multiple classification algorithms, model evaluation, final model selection, Streamlit application development, GitHub version control, and cloud deployment.

The developed application provides a simple and interactive way for users to enter applicant information and obtain a Machine Learning based loan approval prediction.

---

# 🔮 Future Scope

* Use larger and more diverse loan datasets
* Perform advanced hyperparameter optimization
* Compare additional Machine Learning algorithms
* Add probability-based risk assessment
* Integrate a database for storing applications
* Add user authentication
* Improve UI and visualization
* Deploy the system as a production-ready application
* Integrate explainable AI to show the factors influencing predictions

---

**Author:** Thamilarasu S K
**Department:** Computer Science and Engineering
**Institution:** Government College of Engineering, Salem
**Date:** August 2026
**Course / Assessment:** Assessment-1 – Machine Learning Project
