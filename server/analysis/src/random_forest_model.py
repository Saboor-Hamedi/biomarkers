from sklearn.ensemble import RandomForestClassifier

def get_random_forest_model():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        max_depth=10,
        min_samples_split=5,
    )
