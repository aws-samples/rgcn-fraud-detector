import numpy as np
import pandas as pd
from fgnn.fraud_detector import FraudRGCN
from sklearn.metrics import roc_auc_score

df_test = pd.read_parquet('/home/ubuntu/rgcn-fraud-detector/data/test_small.parquet')
df_train = pd.read_parquet('/home/ubuntu/rgcn-fraud-detector/data/train_small.parquet')

### train in inductive mode
fd=FraudRGCN()
fd.train_fg(df_train, params={'n_epochs': 15, 'lr': 0.001})
fd.save_fg("./model/")

fd = FraudRGCN.load_fg('./model')
fraud_proba=fd.predict(df_test, k=2)

fraud_proba1=fd.predict(df_test.head(500), k=2)
fraud_proba2=fd.predict(df_test.tail(500), k=2)

print(f"ROC-AUC inductive (single batch): {roc_auc_score(df_test.isFraud, fraud_proba)}")
print(f"ROC-AUC inductive (two batches): {roc_auc_score(df_test.isFraud, np.concatenate([fraud_proba1, fraud_proba2]))}")

fd=FraudRGCN()
fd.train_fg(pd.concat([df_train, df_test], ignore_index=True),
            params={'n_epochs': 15, 'lr': 0.001},
            test_mask=[False]*len(df_train) + [True]*len(df_test))

fd.save_fg("./model_transductive/")

fd = FraudRGCN.load_fg('./model_transductive')
fraud_proba=fd.predict(df_test)

print(f"ROC-AUC transductive: {roc_auc_score(df_test.isFraud, fraud_proba)}")
