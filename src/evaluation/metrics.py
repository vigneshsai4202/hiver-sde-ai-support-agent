from sklearn.metrics import accuracy_score, f1_score, classification_report

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, zero_division=0)
    }
