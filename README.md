# 🎯 Linear Regression Master

<p align="center">
  <img src="assets/app_screenshot.png" alt="App Screenshot" width="80%">
</p>

[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)


# 🎯 Linear Regression Master — Learn & Explore ML Interactively

Welcome to **Linear Regression Master**, an advanced and beautifully interactive machine learning app built using **Streamlit**. This tool is designed not just to run linear regression models, but to **teach you how they work** — visually, intuitively, and in real-time.

Whether you're a student learning ML, a data science enthusiast, or just curious about how a straight line can predict things — this app is for you.

---

## 🚀 Live App

**🔗 [Try the App Here](https://linear-regression-master.streamlit.app/)**  
> *(Replace the link above with your actual Streamlit app URL after deployment)*

---

## 🧠 What This App Teaches You

- **What is Linear Regression?**
- How slope (`weight`) and intercept (`bias`) affect predictions
- The role of **loss functions** (MSE, MAE, RMSE, R²)
- What are **residuals** and how to minimize them
- How to **optimize parameters** interactively
- Visualizing the **loss landscape in 3D**

---

## 🛠️ Features

✅ Upload your own CSV datasets (2 columns)  
✅ Choose from 4 built-in sample datasets  
✅ Recalculate dynamic parameter ranges  
✅ One-click optimal parameter finder  
✅ Visualizations: regression line, residuals, actual vs predicted  
✅ Real-time interactive feedback on fit score  
✅ Gorgeous custom CSS + animations  
✅ 3D Loss Surface visualization with Plotly  
✅ Beginner-friendly explanations built-in

---

## 📊 Built-in Sample Datasets

| Dataset Name                | Description                        |
|----------------------------|------------------------------------|
| Perfect Linear             | A perfect line: y = 2x             |
| Housing Prices             | Home size vs. price                |
| Study Hours vs Scores      | Hours studied vs. test score       |
| Temperature vs Ice Cream   | Temp vs. ice cream sales           |

You can also upload your own 2-column dataset using the sidebar.

---

## 🧰 Technologies Used

| Library           | Purpose                                     |
|------------------|---------------------------------------------|
| `streamlit`      | Web-based UI for interaction                |
| `pandas`         | Data manipulation                           |
| `numpy`          | Numerical computations                      |
| `matplotlib`     | (Imported but not used – can be removed)    |
| `plotly`         | Dynamic and interactive graphs              |
| `scikit-learn`   | ML models and evaluation metrics            |
| `io`             | Handling CSV uploads                        |
| `time`           | Animation & sleep logic                     |

---

## ⚙️ How to Run This Locally

Clone the repository and run it using Streamlit:

```bash
# Clone this repo
git clone https://github.com/your-username/linear-regression-master-app.git
cd linear-regression-master-app

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
