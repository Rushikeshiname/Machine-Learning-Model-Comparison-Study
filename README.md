# 🤖 Machine Learning Model Comparison Study

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive web application for comparing multiple machine learning classification models on medical datasets. Built with Streamlit, this project provides comprehensive model evaluation using standard metrics and visualizations.

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Models Implemented](#-models-implemented)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Results & Insights](#-results--insights)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Problem Statement

<<<<<<< HEAD

Project Structure-
Machine-Learning-Model-Comparison-Study/

├── model/                          # Folder ( contains trained models)
│
├── .DS_Store                       # Mac system file (can be ignored)
│
├── README.md                       # Project documentation
│
├── app.py                          # Streamlit web application
│
├── heart_disease_uci.csv          # Dataset file
│
└── requirements.txt  

c. Models Used 
The following six machine learning models were implemented and evaluated:
Logistic Regression
Decision Tree
k-Nearest Neighbors (kNN)
Naive Bayes
Random Forest (Ensemble)
XGBoost (Ensemble)
Each model was trained on the same training data and evaluated using identical performance metrics.
=======
The objective of this project is to perform a **comparative analysis of multiple machine learning classification models** on a medical dataset. The goal is to:
>>>>>>> 19bad69 (Update readme file)

- Evaluate how different algorithms perform in terms of predictive accuracy and robustness
- Use standard evaluation metrics: **Accuracy, AUC, Precision, Recall, F1-score, and MCC**
- Identify the most suitable model for the chosen dataset
- Understand the trade-offs between different learning approaches

This comparison helps data scientists and healthcare professionals make informed decisions when selecting classification models for medical diagnosis tasks.

## 📊 Dataset Description

**Dataset:** Heart Disease UCI  
**Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

### Key Characteristics

- **Type:** Supervised Classification Dataset
- **Features:** Multiple numerical and categorical features
- **Target Variable:** Binary classification (presence/absence of heart disease)
- **Size:** 303 instances with 14 attributes

### Preprocessing Steps

<<<<<<< HEAD
Conclusion
From the comparative study, Logistic Regression emerged as the best-performing model in terms of overall accuracy and MCC, while ensemble methods such as Random Forest and XGBoost demonstrated stable and reliable performance.
This study highlights the importance of evaluating multiple metrics rather than relying on accuracy alone when selecting a classification model.


=======
1.  Handling missing values
2. Feature scaling using StandardScaler
3. Label encoding for categorical variables
4. Train-test split for unbiased evaluation

## 🤖 Models Implemented

The following **six machine learning models** were implemented and evaluated:

1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Non-linear tree-based model
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Naive Bayes** - Probabilistic classifier
5. **Random Forest** - Ensemble learning (Bagging)
6. **XGBoost** - Gradient boosting ensemble

Each model was trained on identical data and evaluated using the same metrics for fair comparison.

## 📁 Project Structure

```
Machine-Learning-Model-Comparison-Study/
│
├── model/                          # Trained model files (.pkl, .joblib)
│
├── .DS_Store                       # Mac system file (ignored)
│
├── README.md                       # Project documentation (this file)
│
├── app.py                          # Streamlit web application
│
├── heart_disease_uci.csv          # Heart Disease UCI dataset
│
└── requirements.txt                # Python dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Rushikeshiname/Machine-Learning-Model-Comparison-Study.git
cd Machine-Learning-Model-Comparison-Study
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Upload Dataset**: Click on the sidebar to upload your CSV file (or use the included `heart_disease_uci.csv`)
2. **Select Model**: Choose a classification model from the dropdown menu
3. **View Results**: Explore evaluation metrics, confusion matrix, and feature importance
4. **Compare Models**: Switch between different models to compare their performance

### Deployed Application

Access the live application here: **[Your Streamlit App URL]**

## 📈 Model Performance

### Evaluation Metrics Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| **Logistic Regression** | 0.6066 | 0.8287 | 0.5617 | 0.6066 | 0.5821 | **0.3637** |
| Decision Tree | 0.4426 | 0.6683 | 0.4474 | 0.4426 | 0.4392 | 0.1609 |
| KNN | 0.5246 | 0.7686 | 0.4419 | 0.5246 | 0.4797 | 0.2091 |
| Naive Bayes | 0.5246 | **0.8389** | 0.5491 | 0.5246 | 0.5348 | 0.2829 |
| Random Forest | 0.5738 | 0.8256 | 0.4774 | 0.5738 | 0.5199 | 0.2826 |
| XGBoost | 0.5246 | 0.7859 | 0.5281 | 0.5246 | 0.5227 | 0.2601 |

### Model Performance Observations

| Model | Key Observations |
|-------|------------------|
| **Logistic Regression** | ✅ Best overall performance with highest accuracy (0.6066) and MCC (0.3637). Strong baseline model with high AUC (0.8287). |
| **Decision Tree** | ❌ Weakest performer with lowest accuracy (0.4426) and MCC (0.1609). Shows signs of overfitting. |
| **KNN** | ⚠️ Moderate performance. Reasonable recall but lower precision indicates limited discriminative power. |
| **Naive Bayes** | ✅ Highest AUC (0.8389) showing excellent class separation despite moderate accuracy. |
| **Random Forest** | ✅ Improved over single Decision Tree. Balanced metrics with good stability from ensemble learning. |
| **XGBoost** | ✅ Competitive and consistent results. Stable performance across all metrics. |

## ✨ Features

### Interactive Dashboard
- 📤 **File Upload**: Support for CSV datasets
- 🎯 **Model Selection**: Dynamic dropdown for choosing ML models
- 📊 **Real-time Training**: Train models on uploaded data
- 📈 **Visualization**: Interactive confusion matrices and charts

### Comprehensive Metrics
- Accuracy, AUC, Precision, Recall, F1-Score, MCC
- Confusion Matrix heatmaps
- Classification reports
- Feature importance plots (for tree-based models)

### User-Friendly Interface
- Clean, modern UI with custom styling
- Responsive design for all screen sizes
- Helpful tooltips and instructions
- Dataset preview and statistics

## 🛠️ Technologies Used

### Core Libraries
- **Streamlit** (1.31.0+) - Web application framework
- **scikit-learn** (1.4.0+) - Machine learning models and metrics
- **XGBoost** (2.0.0+) - Gradient boosting implementation

### Data Processing
- **pandas** (2.2.0+) - Data manipulation
- **numpy** (1.26.0+) - Numerical computing

### Visualization
- **matplotlib** (3.8.0+) - Plotting library
- **seaborn** (0.13.0+) - Statistical visualizations

### Model Persistence
- **joblib** (1.3.0+) - Model serialization

## 🔍 Results & Insights

### Key Findings

1. **Logistic Regression** emerged as the best-performing model with:
   - Highest accuracy (60.66%)
   - Highest MCC (0.3637)
   - Strong AUC performance (0.8287)

2. **Naive Bayes** showed the highest AUC (0.8389), indicating excellent class separation despite moderate overall accuracy.

3. **Ensemble methods** (Random Forest and XGBoost) demonstrated stable and reliable performance, validating the power of ensemble learning.

4. **Decision Tree** performed poorly, highlighting the importance of ensemble techniques to reduce overfitting.

### Important Takeaways

✅ **Multiple metrics matter** - Relying on accuracy alone can be misleading  
✅ **Ensemble methods** provide more stable predictions  
✅ **Model selection** depends on the specific use case and metric priorities  
✅ **Trade-offs exist** between different performance aspects

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Rushikesh Iname**

- GitHub: [@Rushikeshiname](https://github.com/Rushikeshiname)
- Project Link: [Machine-Learning-Model-Comparison-Study](https://github.com/Rushikeshiname/Machine-Learning-Model-Comparison-Study)

---

## 🙏 Acknowledgments

- Dataset provided by [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- Available on [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Built with [Streamlit](https://streamlit.io/)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by Rushikesh Iname

</div>
>>>>>>> 19bad69 (Update readme file)
