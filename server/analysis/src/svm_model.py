from sklearn.svm import SVC

def get_svm_model():
    return SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight="balanced",
        C=1.0,
        gamma="scale",
    )
