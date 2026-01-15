# 🌦️ SkyCast AI — Predictive Weather Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green?style=for-the-badge" alt="ML">
  <img src="https://img.shields.io/badge/Dash-Plotly-orange?style=for-the-badge&logo=plotly" alt="Dash">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

**SkyCast AI** is a production-grade Machine Learning ecosystem designed to predict Australian rainfall. By combining advanced feature engineering with a Random Forest architecture, it transforms raw meteorological data into actionable insights through a dual-interface system: an analytical dashboard and a real-time prediction app.

---

## 🎯 Project Objective
* **Predict:** High-accuracy classification of `RainTomorrow`.
* **Resolve:** Advanced handling of class imbalance and missing environmental data.
* **Visualize:** Dynamic exploration of seasonal trends and wind patterns.
* **Deploy:** Professional UI/UX using Glassmorphism and animated components.

---

## 🏗️ System Architecture

The ecosystem follows a modular design, decoupling data logic from the user interface to ensure scalability and ease of maintenance.

### ⚙️ Phase 1: Data Engineering (`data_processor.py`)
> **Focus:** Robust cleaning and spatial feature preservation.

* **Hybrid Imputation:** A two-tier strategy using **Time-series (f-fill/b-fill)** for temporal data and **KNN Imputer** for complex environmental correlations.
* **Cyclic Encoding:** Wind directions are transformed into **Sine/Cosine components** to preserve the circular nature of compass degrees.
* **Feature Engineering:** Engineered metrics like `Pressure_Diff` (diurnal change) and `Cloud_Total` to boost predictive signals.

### 🧠 Phase 2: Model Pipeline (`train.py`)
> **Focus:** Classification performance on imbalanced datasets.

* **Algorithm:** **Random Forest Classifier** selected for its high variance handling.
* **Imbalance Handling:** Utilized `class_weight='balanced'` to effectively capture rare rain events.
* **Serialization:** All artifacts (Scalers, Encoders, Imputers) are exported to the `/models` directory as `.pkl` files for instant inference.

### 📊 Phase 3: Deployment Interfaces
| Interface | Tech Stack | Key Functionality |
| :--- | :--- | :--- |
| **Analytics Dashboard** | `Dash` + `Plotly` | Visualizing correlations, Wind Roses, and seasonal trends. |
| **Prediction Web App** | `Streamlit` | Interactive interface for instant weather predictions and feature analysis. |

---

## 🚀 Getting Started

### 1️⃣ Clone & Install
<pre>
git clone [https://github.com/Emanorabi254/EndToEnd-Weather-ML.git)
cd EndToEnd-Weather-ML
pip install -r requirements.txt
</pre>
---

## 2️⃣ The Workflow

Follow these steps in order to prepare the environment:

| Step | Task | Command | Result |
| :--- | :--- | :--- | :--- |
| 1️⃣ | **Prep** | `python prepare_data.py` | Cleaned dataset for Dash |
| 2️⃣ | **Train** | `python train.py` | Saved `.pkl` model artifacts |
| 3️⃣ | **Explore** | `python dashboard_dash.py` | Launch Analytics at `localhost:8050` |
| 4️⃣ | **Predict** | `streamlit run app_streamlit.py` | Launch Web App (Streamlit) |


---

## 📂 Project Structure

| File/Folder | Description |
| :--- | :--- |
| 📁 **models/** | Serialized ML artifacts (.pkl) |
| 📄 **app_streamlit.py** | Streamlit UI (Glassmorphism) |
| 📊 **dashboard_dash.py** | Dash/Plotly analytical tool |
| ⚙️ **data_processor.py** | Core preprocessing & engineering |
| 🧠 **train.py** | Model training script |
| 🛠️ **prepare_data.py** | Initial data cleaning |
| 📋 **requirements.txt** | Project dependencies |
| ☁️ **weatherAUS.csv** | Raw dataset |

---

## ✨ Key Features
* ✅ **Cyclic Wind Encoding:** Real-world realism for directional data.
* ✅ **Hybrid Imputation:** Combines Median, Time-Series, and KNN for 0% data loss.
* ✅ **Explainable AI:** Visual analytics to understand "why" the model predicts rain.
* ✅ **Modern UX:** Streamlit app featuring animated gradients and clean tab-based inputs.

---

## 🔮 Future Roadmap
- [ ] **Hyperparameter Tuning:** Implementing Optuna for RF optimization.
- [ ] **Boosting Models:** Comparative analysis with XGBoost and LightGBM.
- [ ] **Model Interpretability:** Integrating SHAP values for local explanations.
- [ ] **Cloud Deployment:** Containerization via Docker for AWS/GCP.

---

## ✉️ Contact

<p align="center">
  <a href="https://www.linkedin.com/in/eman-orabi254/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  &nbsp; 
  <a href="emanorabi254@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail">
  </a>
</p>
