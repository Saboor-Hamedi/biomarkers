from xgboost import XGBClassifier

def get_xgboost_model(y_train):
    return XGBClassifier(
        n_estimators=100,
        random_state=42,
        scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
        eval_metric="logloss",
        use_label_encoder=False,
        max_depth=6,
        learning_rate=0.1,
    )
