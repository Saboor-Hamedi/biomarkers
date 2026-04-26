from sklearn.linear_model import LogisticRegression

def get_logistic_model():
    return LogisticRegression(
        random_state=42, max_iter=1000, class_weight="balanced", C=1.0
    )
