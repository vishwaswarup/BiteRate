# Biterate

Biterate is a restaurant review sentiment analysis project that leverages natural language processing and machine learning techniques to classify customer reviews as positive or negative. This README provides detailed information about the dataset, preprocessing steps, feature extraction, models trained, sample results, and instructions on how to run the project.

## Dataset Source

The dataset used in this project is sourced from Kaggle and contains thousands of restaurant reviews labeled with sentiment polarity (positive or negative). The reviews include textual feedback from customers that serve as input for sentiment classification.

You can find the original dataset here: [Kaggle Restaurant Reviews Dataset](https://www.kaggle.com/datasets/ahmedabelal/restaurant-reviews)

## Preprocessing

The raw text data undergoes several preprocessing steps to prepare it for machine learning modeling:

- **Tokenization:** Splitting the reviews into individual words or tokens.
- **Lemmatization:** Reducing words to their base or dictionary form to normalize variations.
- **Stopwords Removal:** Eliminating common words that do not contribute to sentiment, such as "the", "and", "is".
- **Negation Handling:** Identifying negation words and modifying subsequent tokens to better capture sentiment flips (e.g., "not good" treated differently from "good").

These preprocessing steps help in cleaning and standardizing the text data for effective feature extraction.

## Feature Extraction

Features are extracted from the preprocessed text using:

- **TF-IDF Vectorization:** Term Frequency-Inverse Document Frequency is used to weigh words according to their importance across the dataset.
- **Bigrams:** In addition to single words (unigrams), pairs of consecutive words are included to capture context and phrases relevant to sentiment.

This approach provides a rich representation of the textual data for model training.

## Models Trained

Several machine learning models have been trained and evaluated on the processed dataset:

- **Naive Bayes:** A probabilistic classifier well-suited for text classification tasks.
- **Logistic Regression:** A linear model that estimates the probability of a review being positive or negative.
- **Random Forest:** An ensemble of decision trees that captures non-linear relationships and interactions among features.

Each model has been tuned and validated using metrics such as accuracy and F1-score to ensure robust performance.

## Sample Results

The models demonstrate strong performance in classifying restaurant review sentiments. For example:

- Naive Bayes achieved an accuracy of 85% and an F1-score of 0.84 on the test set.
- Logistic Regression improved results with an accuracy of 88% and an F1-score of 0.87.
- Random Forest showed comparable performance with an accuracy of 87% and an F1-score of 0.86.

These results highlight the effectiveness of machine learning techniques in understanding customer sentiment from text data.

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
   Download the restaurant reviews dataset from Kaggle and place it in the `data/` directory.

4. **Preprocess the data:**
   Run the preprocessing script to clean and prepare the text data:
   ```bash
   python preprocess.py
   ```

5. **Train models:**
   Train the desired models using:
   ```bash
   python train.py --model naive_bayes
   python train.py --model logistic_regression
   python train.py --model random_forest
   ```

6. **Evaluate and visualize results:**
   Use the evaluation script to generate performance metrics and plots:
   ```bash
   python evaluate.py
   ```

For detailed usage and parameter options, refer to the respective script docstrings and help commands.

---

Thank you for exploring Biterate! Contributions and feedback are welcome.
