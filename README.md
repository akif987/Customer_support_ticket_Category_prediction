# Customer Support Ticket Classifier

## Train and compare models

From the repository root, run:

```text
python src/train.py
```

The training script evaluates Logistic Regression, Multinomial Naive Bayes,
Linear SVM, Random Forest, and Gradient Boosting on the same stratified test
split. It saves the best model by accuracy as `model/classifier.pkl` and also
saves each model separately:

- `model/logistic_regression.pkl`
- `model/multinomial_naive_bayes.pkl`
- `model/linear_svm.pkl`
- `model/random_forest.pkl`
- `model/gradient_boosting.pkl`

Shared and comparison outputs are saved as:

- `model/tfidf_vectorizer.pkl`
- `model/model_comparison.csv`
- `screenshots/model_comparison.png`
- `screenshots/confusion_matrix.png`

## Run the application

After training, start the Streamlit application:

```text
streamlit run src/app.py
```
