# 🛍️ E-commerce Purchase Predictor

A sleek, beginner-friendly Streamlit web application designed to predict the likelihood of a customer making a purchase based on their browsing behavior and session data. 

## 🌟 Features

*   **Interactive UI:** Intuitive sliders and radio buttons for inputting customer details such as Age, Session Duration, Pages Viewed, Items in Cart, Days Since Last Visit, and Discount usage.
*   **Multi-Model Selection:** Choose between three different machine learning models to run predictions:
    *   Random Forest (Recommended)
    *   Decision Tree
    *   Logistic Regression
*   **Real-time Inference:** Instantly view whether a customer is likely to purchase (YES/NO) along with the model's confidence percentage.
*   **Data Summary:** Review the exact input parameters passed to the model in a clean data frame format.

## 🛠️ Technology Stack

*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Data Handling:** [Pandas](https://pandas.pydata.org/)
*   **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (DecisionTreeClassifier, RandomForestClassifier, LogisticRegression)
*   **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed on your system. It is recommended to use a virtual environment.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/purchasepred-app.git](https://github.com/YOUR-USERNAME/purchasepred-app.git)
    cd purchasepred-app
    ```

2.  **Install dependencies:**
    The required packages are listed in `packages.txt` (or a `requirements.txt` file).
    ```bash
    pip install streamlit pandas scikit-learn joblib
    ```

3.  **Ensure Model Files are Present:**
    The application expects the following pre-trained model files in the root directory:
    *   `rf_model.pkl`
    *   `dt_model.pkl` 
    *   `lr_model.pkl`

### Running the App

Execute the following command in your terminal:

```bash
streamlit run app.py
