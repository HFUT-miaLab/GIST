

def compute_TP_FP_TN_FN(y, pred_labels):
    assert len(y) == len(pred_labels)

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == 1:
            if pred_labels[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if pred_labels[i] == 0:
                TN += 1
            else:
                FP += 1
    return TP, FP, TN, FN


def compute_specificity(TN, FP):
    return float(TN) / (TN + FP)


def compute_sensitivity(TP, FN):
    return float(TP) / (TP + FN)


