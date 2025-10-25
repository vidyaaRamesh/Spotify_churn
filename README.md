# 🎵 Spotify User Churn Prediction

A comprehensive machine learning project that predicts whether a Spotify user is likely to cancel their subscription (churn) based on their listening behavior, demographics, and subscription details.

## 📋 Project Overview

This project uses machine learning techniques to analyze user behavior patterns and predict churn probability for Spotify users. The solution includes data analysis, model training using neural networks, and a user-friendly Streamlit web application for real-time predictions.

## 🎯 Features

- **Comprehensive Data Analysis**: Exploratory data analysis of user behavior patterns
- **Neural Network Model**: Deep learning model built with TensorFlow/Keras for accurate predictions
- **Interactive Web App**: Streamlit-based interface for real-time churn prediction
- **Feature Engineering**: Advanced preprocessing including data transformations and encoding
- **Model Persistence**: Saved trained model and preprocessor for deployment

## 📊 Dataset Features

The model considers the following user attributes:

- **Demographics**: Age, Gender, Country
- **Subscription**: Subscription Type (Free, Premium, Family, Student)
- **Device Usage**: Device Type (Desktop, Web, Mobile)
- **Listening Behavior**: 
  - Listening Time (hours)
  - Songs Played Per Day
  - Skip Rate
  - Offline Listening Status
- **Advertising**: Ads Listened Per Week

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vidyaaRamesh/Spotify_churn.git
   cd spotify_churn_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 🔧 Project Structure

```
spotify_churn_prediction/
├── app.py                           # Streamlit web application
├── Spotify_churn_prediction.ipynb   # Jupyter notebook with full analysis
├── spotify_churn_dataset.csv        # Training dataset
├── spotify_churn_best_ann_model.h5  # Trained neural network model
├── preprocessor.pkl                 # Fitted data preprocessor
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## 🧠 Model Architecture

- **Algorithm**: Artificial Neural Network (ANN)
- **Framework**: TensorFlow/Keras
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
- **Architecture**: Multi-layer neural network with dropout for regularization
- **Evaluation**: Accuracy, Precision, Recall, F1-Score metrics

## 📈 Usage

### Web Application

1. Launch the app using `streamlit run app.py`
2. Input user details in the sidebar:
   - Personal information (age, gender, country)
   - Subscription details
   - Listening behavior metrics
3. Click "Predict Churn" to get the prediction
4. View the churn probability and recommendation

### Jupyter Notebook

Explore the complete analysis in `Spotify_churn_prediction.ipynb`:
- Data exploration and visualization
- Feature engineering and preprocessing
- Model training and evaluation
- Performance metrics and analysis

## 🎯 Model Performance

The trained model achieves:
- **High Accuracy**: Optimized neural network architecture
- **Balanced Predictions**: Handles both churn and non-churn cases effectively
- **Feature Importance**: Identifies key factors influencing churn

## 🛠️ Technical Details

### Data Preprocessing
- **Categorical Encoding**: OneHotEncoder for categorical variables
- **Numerical Scaling**: StandardScaler for numerical features
- **Feature Transformation**: Log transformation for skewed features
- **Train-Test Split**: Proper data splitting for model validation

### Model Training
- **Early Stopping**: Prevents overfitting during training
- **Dropout Layers**: Regularization for better generalization
- **Optimization**: Adam optimizer for efficient training
- **Cross-Validation**: Robust model evaluation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Pranav** - Data Science Project

## 🙏 Acknowledgments

- Spotify for inspiring the use case
- The data science community for valuable insights
- TensorFlow and Streamlit teams for excellent frameworks

## 📞 Contact

For questions or suggestions, please reach out through:
- GitHub Issues
- Project Repository: [Spotify_churn](https://github.com/vidyaaRamesh/Spotify_churn)

---

*This project demonstrates the practical application of machine learning in understanding user behavior and predicting business-critical metrics like customer churn.*
