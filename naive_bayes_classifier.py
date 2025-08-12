import numpy as np

# Training data
# Labels: 1 = spam, 0 = not spam
training_data = [
    {"X": [1, 0, 1, 0], "Y": 1},  # "free offer"
    {"X": [1, 1, 0, 0], "Y": 1},  # "win free"
    {"X": [1, 1, 1, 0], "Y": 1},  # "free offer win"
    {"X": [0, 0, 0, 1], "Y": 0},  # "meeting"
    {"X": [0, 0, 1, 1], "Y": 0}   # "meeting offer"
]

# Vocabulary: ["free", "win", "offer", "meeting"]
n_words = 4

# Estimate prior probabilities
n_spam = sum(1 for d in training_data if d["Y"] == 1)
n_non_spam = sum(1 for d in training_data if d["Y"] == 0)
total_emails = len(training_data)

p_y_spam = (n_spam + 1) / (total_emails + 2)  # Laplace smoothing for binary Y
p_y_non_spam = (n_non_spam + 1) / (total_emails + 2)

# Estimate likelihood probabilities with Laplace smoothing
# Denominator adjustment: total emails in class + vocabulary size
word_counts_spam = np.zeros(n_words)
word_counts_non_spam = np.zeros(n_words)

for data in training_data:
    if data["Y"] == 1:
        word_counts_spam += np.array(data["X"])
    else:
        word_counts_non_spam += np.array(data["X"])

p_x_given_y_spam = (word_counts_spam + 1) / (n_spam + n_words)
p_x_given_y_non_spam = (word_counts_non_spam + 1) / (n_non_spam + n_words)

# Function to predict class for a new email
def predict_naive_bayes(email_x):
    # Numerator for Y = 1
    p_spam = p_y_spam
    p_non_spam = p_y_non_spam
    for i in range(n_words):
        p_spam *= p_x_given_y_spam[i] if email_x[i] == 1 else (1 - p_x_given_y_spam[i])
        p_non_spam *= p_x_given_y_non_spam[i] if email_x[i] == 1 else (1 - p_x_given_y_non_spam[i])
    
    # Normalize to get posterior probabilities
    total = p_spam + p_non_spam
    p_spam_normalized = p_spam / total
    p_non_spam_normalized = p_non_spam / total
    
    return 1 if p_spam_normalized > 0.5 else 0, p_spam_normalized, p_non_spam_normalized

# Test with new emails
new_email_1 = [1, 1, 0, 0]  # "free win"
new_email_2 = [0, 0, 1, 1]  # "meeting offer"

prediction_1, p_spam_1, p_non_spam_1 = predict_naive_bayes(new_email_1)
prediction_2, p_spam_2, p_non_spam_2 = predict_naive_bayes(new_email_2)

print(f"Email 1 ('free win') Prediction: {'Spam' if prediction_1 == 1 else 'Not Spam'}, "
      f"P(Spam) = {p_spam_1:.3f}, P(Not Spam) = {p_non_spam_1:.3f}")
print(f"Email 2 ('meeting offer') Prediction: {'Spam' if prediction_2 == 1 else 'Not Spam'}, "
      f"P(Spam) = {p_spam_2:.3f}, P(Not Spam) = {p_non_spam_2:.3f}")
