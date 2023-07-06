from sklearn.metrics import roc_auc_score, f1_score

def get_results(df, y_actual, col):

    roc_number = roc_auc_score(df[y_actual], df[col])

    F1 = f1_score(df[y_actual], df[col])

    return (roc_number,F1)