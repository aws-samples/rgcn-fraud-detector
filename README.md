# RGCN Fraud Detector

## Install Prerequisites

`pip install -r requirements.txt`

## Model Class

`FraudRGCN` class in [`fgnn/fraud_detector.py`](fgnn/fraud_detector.py) 
implements model for supervised fraud detection using Relational 
Graph Convolutional Network (RGCN). The model is implemented with PyTorch
and Deep Graph Library. Class methods support full lifecycle of the
model:

* `train_fg(train_transactions, params=None, test_mask=None)` Model training. Trains RGCN model on
transactions in `train_transactions` DataFrame. Default model parameters can be overloaded
by passing them in a dictionary `params`. Model parameters are described below. 
When array `test_mask` is passed, it is used to identify test transactions in `train_transactions`, 
and the model will be trained in transductive mode by masking out fraud labels of test transactions 
identified with `True` values in `test_mask`. 


* `predict(test_transactions, k=2)` Model inference. Returns fraud probabilities
for transactions in `test_transactions` DataFrame. Parameter `k` is passed to
`dgl.khop_out_subgraph` and controls number of hops used when extracting subgraph
around target nodes from `test_transactions`.


* `save_fg(model_dir)` Model serialization. Saves model to a directory.
 

* `load_fg(model_dir)` Model deserialization. Loads model from a directory.

### Model Parameters

The following model parameters control graph construction from a DataFrame with transactions:

* `target_col` column name to be used to create target nodes in the graph
* `label_col` column name with binary labels for target nodes (1=fraud, 0=legit) 
* `node_cols` comma-separated list of columns that will be used to create entity (non-target) nodes in the graph
* `cat_cols` comma-separated list of columns that will be used as categorical features of target nodes
* `num_cols` comma-separated list of columns that will be used as numerical features of target nodes
 
#### Default Model Parameters
```python
{
    'num_gpus': 0, # 1=gpu, 0=cpu
    'embedding_size': 128,  # size of node embeddings
    'n_layers': 2,  # number of graph layers
    'n_epochs': 50,  # number of training epochs
    'n_hidden': 16,  # number of hidden units
    'dropout': 0.2,  # dropout rate
    'weight_decay': 5e-6,  # L2 penalization term
    'lr': 1e-2,  # learning rate
    'target_col': 'TransactionID',  # target (transaction-id) column
    'node_cols': 'card1,card2,card3,card4,card5,card6,ProductCD,addr1,addr2,P_emaildomain,R_emaildomain',  # columns to create nodes
    'label_col': 'isFraud',  # label column
    # categorical feature columns:
    'cat_cols': 'M1,M2,M3,M4,M5,M6,M7,M8,M9,DeviceType,DeviceInfo,id_12,id_13,id_14,id_15,id_16,id_17,id_18,id_19,id_20,id_21,id_22,id_23,id_24,id_25,id_26,id_27,id_28,id_29,id_30,id_31,id_32,id_33,id_34,id_35,id_36,id_37,id_38',
    # numerical feature columns:
    'num_cols': 'TransactionAmt,dist1,dist2,id_01,id_02,id_03,id_04,id_05,id_06,id_07,id_08,id_09,id_10,id_11,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,V29,V30,V31,V32,V33,V34,V35,V36,V37,V38,V39,V40,V41,V42,V43,V44,V45,V46,V47,V48,V49,V50,V51,V52,V53,V54,V55,V56,V57,V58,V59,V60,V61,V62,V63,V64,V65,V66,V67,V68,V69,V70,V71,V72,V73,V74,V75,V76,V77,V78,V79,V80,V81,V82,V83,V84,V85,V86,V87,V88,V89,V90,V91,V92,V93,V94,V95,V96,V97,V98,V99,V100,V101,V102,V103,V104,V105,V106,V107,V108,V109,V110,V111,V112,V113,V114,V115,V116,V117,V118,V119,V120,V121,V122,V123,V124,V125,V126,V127,V128,V129,V130,V131,V132,V133,V134,V135,V136,V137,V138,V139,V140,V141,V142,V143,V144,V145,V146,V147,V148,V149,V150,V151,V152,V153,V154,V155,V156,V157,V158,V159,V160,V161,V162,V163,V164,V165,V166,V167,V168,V169,V170,V171,V172,V173,V174,V175,V176,V177,V178,V179,V180,V181,V182,V183,V184,V185,V186,V187,V188,V189,V190,V191,V192,V193,V194,V195,V196,V197,V198,V199,V200,V201,V202,V203,V204,V205,V206,V207,V208,V209,V210,V211,V212,V213,V214,V215,V216,V217,V218,V219,V220,V221,V222,V223,V224,V225,V226,V227,V228,V229,V230,V231,V232,V233,V234,V235,V236,V237,V238,V239,V240,V241,V242,V243,V244,V245,V246,V247,V248,V249,V250,V251,V252,V253,V254,V255,V256,V257,V258,V259,V260,V261,V262,V263,V264,V265,V266,V267,V268,V269,V270,V271,V272,V273,V274,V275,V276,V277,V278,V279,V280,V281,V282,V283,V284,V285,V286,V287,V288,V289,V290,V291,V292,V293,V294,V295,V296,V297,V298,V299,V300,V301,V302,V303,V304,V305,V306,V307,V308,V309,V310,V311,V312,V313,V314,V315,V316,V317,V318,V319,V320,V321,V322,V323,V324,V325,V326,V327,V328,V329,V330,V331,V332,V333,V334,V335,V336,V337,V338,V339',
    'class_weight': 1.  # class weight for fraud label, 1/class_weight will be used as weight for legit label
}
```



## Model Evaluation

The model is evaluated on the 
[IEEE CIS fraud dataset](https://www.kaggle.com/c/ieee-fraud-detection/data) from
the Kaggle competition. The evaluation code is organized in the following three notebooks:

* [`01-Prepare-Data.ipynb`](01-Prepare-Data.ipynb) Download the dataset, and create training and test splits. 
We only use competition's training data in our evaluation, since only these transactions have fraud labels.
We sort transactions by timestamp (`TransactionDT`) column and use 80% of transactions for training, while
the last 20% of transactions are used for testing.


* [`02-Train-Fraud-RGCN.ipynb`](02-Train-Fraud-RGCN.ipynb) Train the model in inductive and transductive modes. We train each model five times.


* [`03-Predict-Fraud-RGCN.ipynb`](03-Predict-Fraud-RGCN.ipynb) Evaluate trained models and plot model performance (ROC AUC) for different 
k-hop values with `k=[1,2,3]`. Also, use inductive models to predict on test transactions in batches of ~1000 transactions,
plot model performance and average latency.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

