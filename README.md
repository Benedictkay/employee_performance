

# 🚀 Employee Performance Prediction Deployment

This repository contains a **Streamlit** web application that predicts employee performance ratings based on key professional and demographic factors. The underlying model was developed using data from **INX Future Inc.** to help organizations make data-driven HR decisions.

## 📊 Project Overview
The goal of this deployment is to provide an interface where HR managers can input employee data and receive an instant performance prediction (Rating 2, 3, or 4).

### Key Features:
* **Top Predictors:** The model focuses on the most impactful factors: Salary Hike Percent, Environment Satisfaction, and Years Since Last Promotion.
* **High Accuracy:** Powered by a **Tuned Random Forest Classifier** achieving **~94% accuracy**.
* **Interactive UI:** Built with Streamlit for a seamless user experience.

---

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** Scikit-learn, Joblib
* **Data Handling:** Pandas, NumPy
* **Deployment:** Streamlit Cloud

---

## 📁 Repository Structure
```text
├── app.py                     # Main Streamlit application code
├── final_employee_model.pkl   # Pre-trained Random Forest model
├── scaler.pkl                 # StandardScaler object for input normalization
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Local Deployment

To run this project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Benedictkay/employee_performance.git
   cd employee_performance
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the App:**
   ```bash
   streamlit run app.py
   ```

---

## 📈 Top 3 Performance Drivers
Based on our Feature Importance analysis, the model primarily looks at:
1.  **Salary Hike Percent:** Recognition through financial growth.
2.  **Environment Satisfaction:** The quality of the workplace culture and resources.
3.  **Years Since Last Promotion:** Career stagnation vs. progression.

---
**Developed by Benedict**