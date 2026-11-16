# Real-Time-NLP-Sentiment-Triage
An NLP-based sentiment analysis project using TF-IDF and Multinomial Naive Bayes to classify product reviews, analyze sentiment probabilities, and generate actionable feedback reports using Python and Scikit-learn.
```markdown
# Real-Time NLP Sentiment Triage

> **Project Status:** Educational prototype. The current notebook uses 12 manually written example reviews, so its predictions are demonstrations—not reliable measurements of real customer opinion.

A beginner-friendly Natural Language Processing (NLP) project that classifies product reviews as **positive** or **negative** and prints a simple report with prediction probabilities, recognized vocabulary terms, and a suggested business action.

---

## Table of Contents
1. [What Does This Project Do?](#1-what-does-this-project-do)
2. [Input and Output](#2-input-and-output)
3. [How the Project Works](#3-how-the-project-works)
4. [Important Code Explained](#4-important-code-explained)
5. [Real-World Use Cases](#5-real-world-use-cases)
6. [Technologies Used](#6-technologies-used)
7. [How to Run](#7-how-to-run)
8. [Current Limitations and Next Improvements](#8-current-limitations-and-next-improvements)
9. [Disclaimer](#9-disclaimer)

---

## 1. What Does This Project Do?

Imagine an online store receives hundreds of product reviews. Reading every review manually takes time. This project demonstrates how a machine learning model can help sort review text by sentiment.

**Example:**
* **Input:** `The battery lasts long and the camera is excellent.`
* **Possible Output:** `POSITIVE`
* **Report Details:** Displays the model's positive and negative probability estimates, terms from the model's vocabulary that appear in the review, and a suggested next step.

> **Note:** The model currently has two learned classes: **positive (`1`)** and **negative (`0`)**. It does not train a separate neutral class.

---

## 2. Input and Output

### Input
The program asks the user to type a product review:

```text
Enter a product review to analyze (or type 'exit' to quit):

```

The input is ordinary text. Type `exit` to stop the loop.

### Output

For each non-empty review, the program prints:

| Field | Description |
| --- | --- |
| **Input Review** | The original text entered by the user. |
| **Predicted Sentiment** | Positive, negative, or the program's uncertain category. |
| **Confidence Split** | The model's estimated positive and negative probabilities. |
| **Matched Keywords** | Vocabulary features present in the processed review. |
| **Business Action** | A demonstration suggestion based on the prediction threshold. |

**Example Format** *(actual percentages depend on the trained model)*:

```text
REAL-TIME SENTIMENT ANALYSIS REPORT
Input Review       : "The product works smoothly and I love it."
Predicted Sentiment: POSITIVE
Confidence Split   : Positive: ...% | Negative: ...%
Matched Keywords   : ...
Business Action    : Route to public testimonials / request app store rating.

```

> **Note on Uncertainty:** The `NEUTRAL / UNCERTAIN` label is a rule-based fallback when neither class reaches 60%. It is not a third class learned from neutral examples. The current code's message mentions out-of-vocabulary terms, but a low-confidence prediction does not prove that unseen words caused the uncertainty.

---

## 3. How the Project Works

1. **Create the dataset:** A small list of example reviews is labelled positive (`1`) or negative (`0`).
2. **Clean the text:** Convert text to lowercase and remove punctuation, numbers, and extra spaces.
3. **Convert words into numbers:** TF-IDF represents text as numerical features. Unigrams are individual words; bigrams are pairs of adjacent words. English stop words are removed.
4. **Split the data:** A stratified holdout split is used for a basic validation report.
5. **Train the validation model:** Multinomial Naive Bayes learns from the training portion and is evaluated on the held-out portion.
6. **Train the demonstration model:** A second Naive Bayes model is fitted on the full small example dataset for interactive predictions.
7. **Analyze new text:** The entered review is cleaned and transformed using the already-fitted TF-IDF vectorizer, then passed to the model.

---

## 4. Important Code Explained

### A. Text Cleaning

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

```

* `lower()` makes uppercase and lowercase versions consistent.
* The regular expression removes characters other than English letters and whitespace.
* The second regular expression replaces repeated whitespace with one space.
* `strip()` removes spaces from the beginning and end.

### B. TF-IDF Feature Extraction

```python
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df['cleaned_text'])

```

TF-IDF converts text into numbers that a machine learning algorithm can process. `ngram_range=(1, 2)` includes individual words and two-word combinations. `fit_transform()` learns the vocabulary and creates the numerical feature matrix.

### C. Train the Naive Bayes Model

```python
prod_clf = MultinomialNB(alpha=0.5)
prod_clf.fit(X, y)

```

`MultinomialNB` is a classification algorithm often used with word-count or TF-IDF text features. `alpha` applies smoothing so features not observed with a class do not automatically receive zero probability. This demonstration model is fitted on all 12 examples.

### D. Predict Probabilities for a New Review

```python
vec = vectorizer.transform([cleaned])
probs = prod_clf.predict_proba(vec)[0]

```

The new review is transformed with the same vectorizer used during training. `predict_proba()` returns the model's estimated probability for each class. The code interprets the two values as negative and positive probabilities because the labels are `0` and `1`.

### E. Apply the Triage Rules

```python
if pos_pct >= 60:
    sentiment = "POSITIVE"
elif neg_pct >= 60:
    sentiment = "NEGATIVE"
else:
    sentiment = "NEUTRAL / UNCERTAIN (Out-of-Vocabulary Terms)"

```

This is a manually chosen threshold rule applied to the model's probabilities. It is not a separately trained neutral classifier. The label and wording should be treated as a prototype business rule.

---

## 5. Real-World Use Cases

Similar sentiment-classification workflows can support:

* **E-commerce:** Sorting product feedback for review by a customer-experience team.
* **Customer support:** Helping prioritize potentially negative feedback for human review.
* **App feedback:** Summarizing positive and negative themes in app-store reviews.
* **Product teams:** Tracking how customer feedback changes after a product update.

> **Production Considerations:** A production system would need a much larger representative dataset, careful testing, monitoring for changing language, privacy controls, and human review for important decisions. It should not automatically issue refunds or publish testimonials based only on this small model.

---

## 6. Technologies Used

* **Language:** Python
* **Data Manipulation:** Pandas and NumPy
* **Text Processing:** Regular expressions (`re`) and TF-IDF Vectorizer
* **Machine Learning:** Scikit-learn (Multinomial Naive Bayes, train-test split, and classification metrics)

---

## 7. How to Run

### Requirements

```bash
pip install numpy pandas scikit-learn jupyter

```

### Steps

1. Clone or download this repository.
2. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run the first notebook cell from top to bottom.
4. Enter a review when prompted.
5. Type `exit` to stop the interactive loop.

> **Note:** The notebook contains two alternative implementations. The first is the interactive triage version; the second is a simpler train-and-predict example. Run one version at a time to avoid confusion.

---

## 8. Current Limitations and Next Improvements

### Current Limitations

* The dataset contains only 12 hand-written examples. It is not large or representative enough for generalization.
* Only positive and negative examples are used for training.
* The validation set is very small, so its metrics can change substantially with different examples.
* The probability values are model estimates and are not necessarily calibrated confidence.
* The matched terms show which vocabulary features are present, not which terms caused the prediction.
* The uncertain label is based on a fixed threshold and needs validation.

### Recommended Next Steps

* Use a larger labelled review dataset.
* Create a genuinely separate neutral class if neutral detection is required.
* Evaluate with a confusion matrix and per-class precision/recall/F1.
* Tune thresholds on validation data.
* Consider packaging the model behind a small API or web interface.

---

## 9. Disclaimer

This project is an educational demonstration of text preprocessing, TF-IDF feature extraction, and Naive Bayes classification. It is not a production customer-support decision system.

```

```
