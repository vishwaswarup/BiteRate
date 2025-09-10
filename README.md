# Biterate

Biterate is a comprehensive project focused on analyzing and predicting Bitcoin price movements using machine learning techniques. This README provides detailed information about the dataset, preprocessing steps, feature extraction, models trained, sample results, and instructions on how to run the project.

## Dataset Source

The dataset used in this project is sourced from Kaggle and contains historical Bitcoin price data along with related market indicators. The data includes timestamped records of Bitcoin prices, trading volumes, and other relevant financial metrics.

You can find the original dataset here: [Kaggle Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

## Preprocessing

The raw dataset undergoes several preprocessing steps to prepare it for machine learning modeling:

- **Data Cleaning:** Handling missing values by interpolation and removing any inconsistent entries.
- **Timestamp Conversion:** Converting timestamps to datetime objects and setting them as the index for time series analysis.
- **Normalization:** Scaling numerical features to a standard range to improve model convergence.
- **Feature Engineering:** Creating additional features such as moving averages, volatility indicators, and lagged price values to capture temporal dependencies.
- **Train-Test Split:** Dividing the dataset into training and testing subsets based on chronological order to prevent lookahead bias.

## Feature Extraction

Feature extraction is crucial for capturing the underlying patterns in Bitcoin price movements. The following features are extracted and used for modeling:

- **Technical Indicators:** Moving averages (e.g., 7-day, 21-day), Relative Strength Index (RSI), Bollinger Bands.
- **Volume Metrics:** Trading volume and volume changes over different periods.
- **Price Derivatives:** Daily returns, price momentum, and volatility measures.
- **Lag Features:** Previous days' closing prices to capture temporal dependencies.

These features collectively provide the models with a rich representation of the market conditions.

## Models Trained

Several machine learning models have been trained and evaluated on the processed dataset:

- **Linear Regression:** A baseline model to capture linear relationships between features and Bitcoin price.
- **Random Forest Regressor:** An ensemble model that captures non-linear patterns and interactions.
- **Gradient Boosting Machines (XGBoost):** A powerful boosting model optimized for predictive accuracy.
- **Long Short-Term Memory (LSTM) Networks:** A recurrent neural network architecture designed to capture sequential dependencies in time series data.

Each model has been tuned and validated using appropriate metrics to ensure robust performance.

## Sample Results

The models demonstrate varying degrees of success in predicting Bitcoin price movements. For example:

- The Random Forest model achieved a Mean Absolute Error (MAE) of 350 USD on the test set.
- The LSTM model showed promising results with improved temporal prediction capabilities, reducing prediction lag.
- Feature importance analysis highlighted the significance of moving averages and volume changes in forecasting.

These results indicate the potential of machine learning techniques in understanding and predicting cryptocurrency markets.

## How to Run the Project

To run the Biterate project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/biterate.git
   cd biterate
   ```

2. **Install dependencies:**
   Ensure you have Python 3.7+ installed. Then install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   Download the Bitcoin historical data from Kaggle and place it in the `data/` directory.

4. **Preprocess the data:**
   Run the preprocessing script to clean and prepare the data:
   ```bash
   python preprocess.py
   ```

5. **Train models:**
   Train the desired models using:
   ```bash
   python train.py --model random_forest
   python train.py --model lstm
   ```

6. **Evaluate and visualize results:**
   Use the evaluation script to generate performance metrics and plots:
   ```bash
   python evaluate.py
   ```

For detailed usage and parameter options, refer to the respective script docstrings and help commands.

---

Thank you for exploring Biterate! Contributions and feedback are welcome.
