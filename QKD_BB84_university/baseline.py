import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_gen import MAIN_CSV
from src.dataset import get_qkd_dataloaders
from src.evaluation import per_attack_recall

np.random.seed(42)
# load dataloaders the same as for the nn for comparison

train_loader, test_loader, scaler = get_qkd_dataloaders(
    csv_path=MAIN_CSV,
    window_size=50,
    batch_size=64
)

# convert train and test data into numpy arrays
X_train = train_loader.dataset.X_seq.numpy()
y_train = train_loader.dataset.y_seq.numpy()

X_test = test_loader.dataset.X_seq.numpy()
y_test = test_loader.dataset.y_seq.numpy()

# flatten data into 2d matrices
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# initialise random forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
random_forest.fit(X_train_flat, y_train)

# initialise logistic regression classifier
logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
logistic_regression.fit(X_train_flat, y_train)

# generate the classification for each (0 or 1)
rf_predictions = random_forest.predict(X_test_flat)
lr_predictions = logistic_regression.predict(X_test_flat)

# generate probability score for ROC-AUC metrics
rf_probs = random_forest.predict_proba(X_test_flat)[:, 1]
lr_probs = logistic_regression.predict_proba(X_test_flat)[:, 1]

# print results
ds = test_loader.dataset
for name, preds, probs in [("Random Forest", rf_predictions, rf_probs),
                           ("Logistic Regression", lr_predictions, lr_probs)]:
    print("\n" + "=" * 40)
    print(f"--- {name} Benchmark Results ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1-Score: {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, probs):.4f}")
    table = per_attack_recall(preds, y_test, ds.attack_type, ds.is_stealthy)
    if len(table):
        print("Recall by attack type:")
        print(table.to_string(index=False, float_format="{:.3f}".format))
print("=" * 40 + "\n")
