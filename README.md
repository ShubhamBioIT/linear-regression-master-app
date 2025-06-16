# 🎯 Linear Regression Master

<p align="center">
  <img src="assets/app_screenshot.png" alt="App Screenshot" width="80%">
</p>

[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## 📌 Project Overview

**Linear Regression Master** is an advanced and interactive Streamlit app designed for beginners and intermediate users to **learn, visualize, and explore Linear Regression** with real-time plots and feedback.

You can upload your own CSV files or choose from sample datasets and play with parameters like **slope** and **intercept** to see how your model performs in real-time with live updates of:
- Regression plots
- Residuals
- Metrics like MAE, MSE, RMSE, R²
- 3D Loss Landscape 🌄

---

## 🚀 Live Demo

🔗 [Click here to open the app in Streamlit](https://share.streamlit.io/your-username/your-repo-name/main/app.py)

---

## ✨ Features

- 📊 Upload or use built-in datasets
- 🔁 Recalculate dynamic parameter ranges
- 🎯 Optimize weights/bias with one click
- 🎮 Real-time visualization and feedback
- 🌄 Interactive 3D Loss Landscape (optional)
- 📚 Beginner-friendly explanations with visuals
- 🎛️ Fully styled and animated UI

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| [Python](https://www.python.org/) | Core language |
| [Streamlit](https://streamlit.io/) | App frontend |
| [Plotly](https://plotly.com/) | Graphs & 3D plots |
| [scikit-learn](https://scikit-learn.org/) | Linear models & metrics |
| [Pandas](https://pandas.pydata.org/) | Data processing |
| [NumPy](https://numpy.org/) | Math & vector ops |

---

## 📂 Sample Datasets Included

| Dataset Name                | Description                        |
|----------------------------|------------------------------------|
| Perfect Linear             | Straight line: y = 2x              |
| Housing Prices             | House size vs price                |
| Study Hours vs Scores      | Study time vs exam score           |
| Temperature vs Ice Cream   | Temperature vs sales               |

You can also upload your own CSV file (2 columns only).

---

## 📸 Screenshot

> ![Screenshot of App](assets/app_screenshot.png)

---

## ⚙️ How to Run Locally

```bash
# Clone this repo
git clone https://github.com/your-username/linear-regression-master-app.git
cd linear-regression-master-app

# Create environment and install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
