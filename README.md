# 🎒 Smart Bag Price Predictor

A modern web application that predicts bag prices based on various features using machine learning. Built with Streamlit and scikit-learn.

## ✨ Features

- **Price Prediction**: Get instant price estimates for bags based on multiple features
- **Market Analysis**: View detailed market insights and price distributions
- **Smart Recommendations**: Receive personalized bag recommendations
- **Interactive UI**: Beautiful dark-themed interface with real-time updates
- **Feature Impact**: Understand what factors influence bag prices the most

## 🚀 Key Components

- **Price Prediction Tab**
  - Input multiple bag features (brand, size, material, etc.)
  - Get instant price predictions
  - View market comparison and price position
  - See feature importance analysis

- **Market Analysis Tab**
  - Price distribution visualization
  - Brand market share analysis
  - Material popularity insights
  - Price range statistics

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bag-price-predictor.git
cd bag-price-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## 📦 Dependencies

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## 💡 Usage

1. Launch the app using `streamlit run app.py`
2. Select the "Price Prediction" tab to estimate bag prices
3. Fill in the bag details:
   - Brand
   - Material
   - Size
   - Color
   - Special features
4. Click "Calculate Price" to get the prediction
5. View the market analysis tab for deeper insights

## 🎯 Model Details

The price prediction model is trained on a dataset of bag prices and features. It uses:
- Random Forest Regressor
- Feature engineering for categorical variables
- Cross-validation for model evaluation

## 📊 Screenshots

[Add screenshots of your application here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
