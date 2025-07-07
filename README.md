![Workshop Banner](https://res.cloudinary.com/dtradpei6/image/upload/data_bfnxm8.jpg)
[![GitHub Pages](https://img.shields.io/badge/View%20Site-GitHub%20Pages-blue?logo=github)](https://santiago2588.github.io/distillation_column_training/)

# Distillation Column Yield Prediction – Workshop 🧪

## Overview 📋
This hands-on workshop guides participants through building and evaluating various regression models to predict the yield in distillation columns. Participants will learn by working through a series of Jupyter notebooks, from data loading and preparation to model training and evaluation.

## What You’ll Learn 🧠
*   Loading, inspecting, and cleaning real-world data.
*   Performing feature engineering to improve model performance.
*   Establishing a baseline model for comparison.
*   Implementing and fine-tuning tree-based regression models:
    *   Random Forest
    *   Gradient Boosting
    *   LightGBM (LGBM)
    *   XGBoost
*   Evaluating model performance using appropriate metrics.
*   Understanding the workflow of a machine learning project from data to model.

## Getting Started 🛠️
✅ **Recommended Platform: Google Colab**
Google Colab provides a free, interactive environment that's ideal for this workshop. No local installation is required!

**What You Need:**
*   A Google account.
*   A reliable internet connection.

**Running the Notebooks in Colab:**
1.  **Access the Notebooks:**
    *   Open the main GitHub repository page for this workshop.
    *   Navigate to the `Soluciones/` directory.
2.  **Open in Colab:**
    *   Click on a notebook file (e.g., `01_load_and_clean_data.ipynb`).
    *   Look for an "Open in Colab" badge/button at the top of the notebook preview on GitHub. Click it.
    *   *Alternatively*, if the badge isn't available:
        *   On the GitHub notebook page, click the "Raw" button. Copy the URL from your browser's address bar.
        *   Open Google Colab (<https://colab.research.google.com/>).
        *   Select `File > Open notebook`.
        *   Choose the "GitHub" tab, paste the URL, and press Enter.
3.  **Install Dependencies (in Colab):**
    *   Once a notebook is open in Colab, the first code cell in many notebooks will be for installing necessary libraries.
    *   If not, you can create a new code cell and run:
        ```python
        !pip install pandas scikit-learn matplotlib seaborn xgboost lightgbm
        ```
    *   Run this cell by pressing Shift+Enter or clicking the play button.

📘 **Colab Tips:**
*   Notebooks are interactive. Run code cells one by one.
*   Save a copy to your Google Drive (`File > Save a copy in Drive`) if you want to save your work and modifications.
*   [Colab FAQ](https://research.google.com/colaboratory/faq.html)

## Workshop Sessions 📚
Each session corresponds to a Jupyter notebook in the `Soluciones/` directory.

| Session | Notebook                                 | Topic                                      | Estimated Duration |
| :------ | :--------------------------------------- | :----------------------------------------- | :----------------- |
| 1       | `01_load_and_clean_data.ipynb`           | Loading, Inspecting & Cleaning Data        | ~1-1.5 hr          |
| 2       | `02_feature_engineering.ipynb`         | Feature Engineering Techniques             | ~1-1.5 hr          |
| 3       | `03_baseline_model.ipynb`                | Creating a Baseline Model                  | ~1 hr              |
| 4       | `04_random_forest_regression.ipynb`      | Random Forest Regression                   | ~1-1.5 hr          |
| 5       | `05_gradient_boosting_regression.ipynb`  | Gradient Boosting Regression             | ~1-1.5 hr          |
| 6       | `06_lgbm_regression.ipynb`               | LightGBM (LGBM) Regression               | ~1-1.5 hr          |
| 7       | `07_xgboost_regression.ipynb`            | XGBoost Regression                         | ~1-1.5 hr          |
| (Optional)| `main.py` (Streamlit app)            | Deploying a Model with Streamlit (Demo)    | ~0.5-1 hr          |

## Learning Outcomes 🎯
By the end of this workshop, you’ll be able to:
*   Confidently load, clean, and prepare data for machine learning tasks.
*   Apply various feature engineering techniques.
*   Build, train, and evaluate several industry-standard regression models in Python.
*   Understand the differences and trade-offs between Random Forest, Gradient Boosting, LGBM, and XGBoost.
*   Follow a structured approach to solving regression problems with machine learning.

## Repository Structure 📁
```
.
├── images/                      # Contains images used in the Streamlit application
│   └── column.jpg
├── Soluciones/                  # Workshop notebooks: from data processing to modeling
│   ├── 01_load_and_clean_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_random_forest_regression.ipynb
│   ├── 05_gradient_boosting_regression.ipynb
│   ├── 06_lgbm_regression.ipynb
│   └── 07_xgboost_regression.ipynb
├── config.yaml                  # Configuration file for file paths (used by notebooks/scripts)
├── data/                        # Contains datasets for the workshop
│   ├── raw_data.csv
│   ├── clean_data.csv
│   ├── transformed_data.csv
│   └── transformed_normalized_data.csv
├── main.py                      # Example Streamlit application script for model deployment
├── model/                       # Contains a pre-trained example model (used by main.py)
│   └── final_model.joblib
└── requirements.txt             # Lists Python dependencies for local setup / Colab
```

## Prerequisites 📾
*   **Basic Python skills:** Familiarity with data types, loops, functions, and basic syntax.
*   **Some knowledge of basic machine learning concepts:** Understanding of terms like features, target, training, testing, and model evaluation.
*   **Familiarity with linear algebra/calculus (optional):** Helpful for a deeper understanding of some algorithms, but not strictly required for completing the workshop.
*   **No Scikit-learn (or other specific library) experience required!** The workshop will guide you through their usage.

## Additional Resources (Optional) 📚
*   [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) by Jake VanderPlas
*   [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
*   [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
*   [LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/)
