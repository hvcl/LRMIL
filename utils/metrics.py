import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(labels, preds, probs, num_classes):
    if len(labels) == 0:
        return {"acc": 0.0, "f1": 0.0, "auc": 0.0}

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    try:
        probs_np = np.vstack(probs)
        labels_np = np.asarray(labels)
        if num_classes == 2:
            auc = roc_auc_score(labels_np, probs_np[:, 1])
        else:
            auc = roc_auc_score(labels_np, probs_np, multi_class="ovr")
    except Exception:
        auc = 0.0
    return {"acc": acc, "f1": f1, "auc": auc}
