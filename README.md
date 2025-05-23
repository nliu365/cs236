Stanford cs236 Deep Generative Models | 2023

* lecture 2
    Naive Bayes for Classification in naive_bayes_classifier.py
  
    ```
    Let’s walk through the Naive Bayes classification process for spam detection step-by-step using the principles from the slide. We'll create a simple example
    with a small vocabulary, train a model, and then use it to predict whether new emails are spam or not spam.


    Step 1: Define the Problem and Training Data
    We’ll classify emails as spam (Y = 1) or not spam (Y = 0). Let’s use a small vocabulary of 4 words: {“free”, “win”, “offer”, “meeting”}. Each X_i is 1 if word
     i appears in the email, and 0 otherwise. We'll create a small training dataset with 5 emails (3 spam, 2 non-spam) represented as binary vectors (X_1, X_2, X_3,
    X_4) corresponding to the words in order.

    Training Data
    Spam Emails (Y = 1):
    * Email 1: “free offer” → (1, 0, 1, 0)
    * Email 2: “win free” → (1, 1, 0, 0)
    * Email 3: “free offer win” → (1, 1, 1, 0)
    
    Non-Spam Emails (Y = 0):
    * Email 4: “meeting” → (0, 0, 0, 1)
    * Email 5: “meeting offer” → (0, 0, 1, 1)


    Step 2: Estimate Parameters from Training Data
    Using the Naive Bayes assumption that words are conditionally independent given Y, we estimate:
    * p(Y = y): Prior probability of spam or not spam.
    * p(X_i = 1 | Y = y): Probability of word i appearing given the email is spam or not spam.

    Prior Probabilities
    * Total emails = 5
    * Spam emails (Y = 1) = 3 → p(Y = 1) = 3/5 = 0.6
    * Non-spam emails (Y = 0) = 2 → p(Y = 0) = 2/5 = 0.4

    Likelihood Probabilities (using Laplace smoothing to avoid zero probabilities)
    For each word X_i and class Y, count the number of emails where X_i = 1 and divide by the total number of emails in that class (with a smoothing factor of 1
    for each word and class to handle unseen data):

    * Number of spam emails = 3
    * Number of non-spam emails = 2
    * Vocabulary size = 4
    | Word    | p(X_i = 1 | Y = 1) (Spam) | p(X_i = 1 | Y = 0) (Not Spam) |
    |---------|-----------------------|---------------------------|
    | free    | (2 + 1) / (3 + 4) = 3/7 ≈ 0.429 | (0 + 1) / (2 + 4) = 1/6 ≈ 0.167 |
    | win     | (2 + 1) / (3 + 4) = 3/7 ≈ 0.429 | (0 + 1) / (2 + 4) = 1/6 ≈ 0.167 |
    | offer   | (2 + 1) / (3 + 4) = 3/7 ≈ 0.429 | (1 + 1) / (2 + 4) = 2/6 = 0.333 |
    | meeting | (0 + 1) / (3 + 4) = 1/7 ≈ 0.143 | (2 + 1) / (2 + 4) = 3/6 = 0.5   |

    (Counts: "free" appears 2 times in spam, 0 in non-spam; "win" appears 2 times in spam, 0 in non-spam; "offer" appears 2 times in spam, 1 in non-spam;
    "meeting" appears 0 times in spam, 2 in non-spam. Laplace smoothing adds 1 to each count and 4 to the denominator.)


    Step 3: Build the Model
    The Naive Bayes model is defined by the prior p(Y) and the likelihoods p(X_i | Y). For prediction, we use Bayes' rule:
    p(Y = 1 | X_1, ..., X_n) = (p(Y = 1) * ∏{i=1}^n p(X_i | Y = 1)) / (∑{y ∈ {0, 1}} p(Y = y) * ∏_{i=1}^n p(X_i | Y = y))


    Step 4: Apply the Model to New Emails
    We’ll test two new emails:

    * New Email 1 (Spam-like): “free win” → (1, 1, 0, 0)
    * New Email 2 (Non-Spam-like): “meeting offer” → (0, 0, 1, 1)

    Prediction for New Email 1: “free win” (X = (1, 1, 0, 0))
    * Numerator (for Y = 1): p(Y = 1) * p(X_1 = 1 | Y = 1) * p(X_2 = 1 | Y = 1) * p(X_3 = 0 | Y = 1) * p(X_4 = 0 | Y = 1) = 0.6 * 0.429 * 0.429 * (1 - 0.429) *
    (1 - 0.143) = 0.6 * 0.429 * 0.429 * 0.571 * 0.857 ≈ 0.0607
    * Denominator (sum over Y = 0 and Y = 1):
      ** For Y = 0: p(Y = 0) * p(X_1 = 1 | Y = 0) * p(X_2 = 1 | Y = 0) * p(X_3 = 0 | Y = 0) * p(X_4 = 0 | Y = 0) = 0.4 * 0.167 * 0.167 * (1 - 0.333) * (1 - 0.5)
    = 0.4 * 0.167 * 0.167 * 0.667 * 0.5 ≈ 0.0037
      ** Total denominator = 0.0607 + 0.0037 ≈ 0.0644
    * Posterior p(Y = 1 | X) = 0.0607 / 0.0644 ≈ 0.943
    * Since p(Y = 1 | X) > 0.5, classify as spam.

    Prediction for New Email 2: “meeting offer” (X = (0, 0, 1, 1))
    * Numerator (for Y = 1): 0.6 * (1 - 0.429) * (1 - 0.429) * 0.429 * 0.143 = 0.6 * 0.571 * 0.571 * 0.429 * 0.143 ≈ 0.0079
    * Denominator:
      ** For Y = 0: 0.4 * (1 - 0.167) * (1 - 0.167) * 0.333 * 0.5 = 0.4 * 0.833 * 0.833 * 0.333 * 0.5 ≈ 0.0463
      ** Total denominator = 0.0079 + 0.0463 ≈ 0.0542
    * Posterior p(Y = 1 | X) = 0.0079 / 0.0542 ≈ 0.146
    * Since p(Y = 1 | X) < 0.5, classify as not spam.
    
    Summary
    The model successfully identifies “free win” as spam and “meeting offer” as not spam, aligning with typical patterns. The independence assumption simplifies
    computation but may not hold perfectly (e.g., “free” and “win” often co-occur in spam), yet it remains useful as noted in the slide’s philosophy.
    ```

Summary of Key Differences
```
Aspect	Step-by-Step Example(naive_bayes_classifier.py)	Python Code Implementation(text_classifier/bayes.py)
Vocabulary	Fixed (4 words)	Dynamic (built from dataset, 32 words)
Smoothing	Laplace smoothing applied	No smoothing, risking zero probabilities
Data Size	Small (5 emails)	Small but scalable (6 documents)
Training	Manual probability calculations	Automated via trainNB0 function
Prediction	Explicitly shown for two test emails	Not implemented in provided code
Normalization	Document-based (per class + vocab size)	Word frequency-based (total word count per class)
Scalability	Low, manual process	High, automated and adaptable
Flexibility	Limited by fixed vocabulary	High, dynamic vocabulary generation
```
