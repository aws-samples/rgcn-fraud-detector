""" RGCN Fraud Detector Model
"""
__author__ = "Dmitriy Bespalov"


import os
import sys
import json

import dgl
import numpy as np
import pandas as pd
import time
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer

try:
    # use relative import when running code inside notebook
    from .pytorch_model import HeteroRGCN
    from .model_train_utils import train_fg, normalize_test, normalize_train, encode_node_ids, x_plus_log10
except ImportError:
    # use this import when running code inside SageMaker estimator
    from pytorch_model import HeteroRGCN
    from model_train_utils import train_fg, normalize_test, normalize_train, encode_node_ids, x_plus_log10

import torch as th

from typing import List, Dict, Any

pd.options.mode.use_inf_as_na = True


class FraudRGCN:

    def __init__(self):
        """
        Constructor for FraudRGCN object
        """
        self._train_g = None
        self._timings = {
            'train: construct graph':[],
            'train: fit model':[],
            'train: total':[],
            'predict: extend graph':[],
            'predict: extract subgraph':[],
            'predict: copy embedding':[],
            'predict: inference':[],
            'predict: total':[],
            'predict: full-graph num nodes': [],
            'predict: sub-graph num nodes': [],
        }

        ### defaul model parameters
        self._default_params = {
            'num_gpus': 0,
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
            # categorical feature columns
            'cat_cols': 'M1,M2,M3,M4,M5,M6,M7,M8,M9,DeviceType,DeviceInfo,id_12,id_13,id_14,id_15,id_16,id_17,id_18,id_19,id_20,id_21,id_22,id_23,id_24,id_25,id_26,id_27,id_28,id_29,id_30,id_31,id_32,id_33,id_34,id_35,id_36,id_37,id_38',
            # numerical feature columns
            'num_cols': 'TransactionAmt,dist1,dist2,id_01,id_02,id_03,id_04,id_05,id_06,id_07,id_08,id_09,id_10,id_11,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,V29,V30,V31,V32,V33,V34,V35,V36,V37,V38,V39,V40,V41,V42,V43,V44,V45,V46,V47,V48,V49,V50,V51,V52,V53,V54,V55,V56,V57,V58,V59,V60,V61,V62,V63,V64,V65,V66,V67,V68,V69,V70,V71,V72,V73,V74,V75,V76,V77,V78,V79,V80,V81,V82,V83,V84,V85,V86,V87,V88,V89,V90,V91,V92,V93,V94,V95,V96,V97,V98,V99,V100,V101,V102,V103,V104,V105,V106,V107,V108,V109,V110,V111,V112,V113,V114,V115,V116,V117,V118,V119,V120,V121,V122,V123,V124,V125,V126,V127,V128,V129,V130,V131,V132,V133,V134,V135,V136,V137,V138,V139,V140,V141,V142,V143,V144,V145,V146,V147,V148,V149,V150,V151,V152,V153,V154,V155,V156,V157,V158,V159,V160,V161,V162,V163,V164,V165,V166,V167,V168,V169,V170,V171,V172,V173,V174,V175,V176,V177,V178,V179,V180,V181,V182,V183,V184,V185,V186,V187,V188,V189,V190,V191,V192,V193,V194,V195,V196,V197,V198,V199,V200,V201,V202,V203,V204,V205,V206,V207,V208,V209,V210,V211,V212,V213,V214,V215,V216,V217,V218,V219,V220,V221,V222,V223,V224,V225,V226,V227,V228,V229,V230,V231,V232,V233,V234,V235,V236,V237,V238,V239,V240,V241,V242,V243,V244,V245,V246,V247,V248,V249,V250,V251,V252,V253,V254,V255,V256,V257,V258,V259,V260,V261,V262,V263,V264,V265,V266,V267,V268,V269,V270,V271,V272,V273,V274,V275,V276,V277,V278,V279,V280,V281,V282,V283,V284,V285,V286,V287,V288,V289,V290,V291,V292,V293,V294,V295,V296,V297,V298,V299,V300,V301,V302,V303,V304,V305,V306,V307,V308,V309,V310,V311,V312,V313,V314,V315,V316,V317,V318,V319,V320,V321,V322,V323,V324,V325,V326,V327,V328,V329,V330,V331,V332,V333,V334,V335,V336,V337,V338,V339',
            'class_weight': 1.  # class weight for fraud label, 1/class_weight will be used as weight for legit label
        }

    def predict(self, test_transactions: pd.DataFrame, k: int = 2):
        """
        FraudRGCN model inference.

        :param test_transactions: DataFrame with transactions to predict fraud
        :param k: number of hops to use when extracting subgraph. Parameter is passed to dgl.khop_out_subgraph
        :return: returns array with predicted fraud probabilities for test_transactions
        """

        if self._train_g is None:
            raise RuntimeError("Model must be trained first!")

        if self._params['num_gpus'] > 0:
            device = th.device('cuda:0')
        else:
            device = th.device('cpu')

        t1 = time.time()

        target_nodes, added_nodes = self._extend_graph(test_transactions,
                                                       self._params['target_col'],
                                                       self._params['node_cols'])

        t2 = time.time()

        test_g, inverse_target_nodes = dgl.khop_out_subgraph(self._train_g, {'target': target_nodes}, k=k)

        test_features = test_g.nodes['target'].data['features']
        test_features = test_features.to(device)

        train_n_nodes = th.sum(th.tensor([self._train_g.number_of_nodes(n_type) for n_type in self._train_g.ntypes]))

        test_n_nodes = th.sum(th.tensor([test_g.number_of_nodes(n_type) for n_type in test_g.ntypes]))
        test_n_edges = th.sum(th.tensor([test_g.number_of_edges(e_type) for e_type in test_g.etypes]))

        print("""----Inference Data statistics------'
                    #Nodes: {}
                    #Edges: {}
                    #Features Shape: {}""".format(test_n_nodes,
                                                  test_n_edges,
                                                  test_features.shape, ))

        t3 = time.time()

        model = self._model
        embed_copy = dict(model.embed)

        print("Starting Model inference")
        for ntype, emb_ in model.embed.items():
            train_num = self._train_g.number_of_nodes(ntype)
            test_num = test_g.number_of_nodes(ntype)

            mean_emb = th.mean(emb_, dim=0)

            new_emb = mean_emb.repeat(test_num, 1).detach().numpy()

            ### for nodes in subgraph, get their node-ids in train_g (full graph)
            train_g_ids = test_g.ndata[dgl.NID][ntype].numpy()

            ### filter out subgraph nodes that were added to train_g after training,
            ### since only these nodes will have learned embedding
            emb_train_g_ids = np.where(train_g_ids<emb_.shape[0])[0]

            print(f"Number of nodes type {ntype}: train={train_num} subgraph={test_num} has_embedding={len(emb_train_g_ids)}")

            ### copy embedding for subgraph nodes that were "seen" during training
            ### and mean-fill other nodes in subgraph
            new_emb[emb_train_g_ids, :] = th.index_select(emb_, 0, th.from_numpy(
                train_g_ids[emb_train_g_ids])).detach().numpy()

            model.embed[ntype] = th.nn.Parameter(th.from_numpy(new_emb))

        t4 = time.time()

        unnormalized_preds = model(test_g, test_features.to(device))
        pred_proba = th.softmax(unnormalized_preds, dim=-1)
        fraud_proba = pred_proba[:, 1].detach().numpy()

        model.embed = th.nn.ParameterDict(embed_copy)

        ### clean-up graph: remove newly added nodes from graph and from node-id lookups
        for ntype, nodes_tup in added_nodes.items():
            new_node_ids, new_node_vals = nodes_tup
            self._train_g.remove_nodes(new_node_ids, ntype=ntype)

            for new_val in new_node_vals:
                del self._nodes_lookup[ntype][new_val]

        t5 = time.time()

        self._timings['predict: full-graph num nodes'].append(train_n_nodes)
        self._timings['predict: sub-graph num nodes'].append(test_n_nodes)

        self._timings['predict: extend graph'].append(t2-t1)
        self._timings['predict: extract subgraph'].append(t3-t2)
        self._timings['predict: copy embedding'].append(t4-t3)
        self._timings['predict: inference'].append(t5-t4)
        self._timings['predict: total'].append(t5-t1)

        return fraud_proba[inverse_target_nodes['target'].numpy()]


    def save_fg(self, model_dir: str):
        """
        Serialize model to directory.

        :param model_dir: path to directory
        :return: None
        """

        if self._train_g is None:
            raise RuntimeError("Model must be trained first!")

        train_g = getattr(self, '_train_g', None)
        model = getattr(self, '_model', None)

        print(f"Saving model to {model_dir}")

        os.makedirs(model_dir, exist_ok=True)

        # save torch model parameters and dgl.heterograph to model.pth
        th.save({"model": model.state_dict(),
                 "train_g": train_g},
                os.path.join(model_dir, 'model.pth'))

        self._model = None
        self._train_g = None

        # save FraudRGCN object (w/o torch model and heterograph object) to fraud_detector.pkl
        with open(os.path.join(model_dir, 'fraud_detector.pkl'), 'wb') as f:
            pickle.dump(self, f)

        self._train_g = train_g
        self._model = model

    @staticmethod
    def load_fg(model_dir: str):
        """
        Load model from serialized state in a directory.

        :param model_dir: path to directory
        :return: returns FraudRGCN object
        """

        print(f"Loading model from {model_dir}")

        model_g = th.load(os.path.join(model_dir, 'model.pth'))

        with open(os.path.join(model_dir, 'fraud_detector.pkl'), 'rb') as f:
            detector = pickle.load(f)

        detector._train_g = model_g['train_g']
        model_dict = model_g['model']
        train_g = model_g['train_g']
        ntype_dict = {n_type: train_g.number_of_nodes(n_type) for n_type in train_g.ntypes}

        in_feats = train_g.nodes['target'].data['features'].shape[1]
        n_classes = 2

        if detector._params['num_gpus'] > 0:
            device = th.device('cuda:0')
        else:
            device = th.device('cpu')

        model = HeteroRGCN(ntype_dict, train_g.etypes, in_feats, detector._params['n_hidden'], n_classes, detector._params['n_layers'], in_feats)
        model = model.to(device)
        model.load_state_dict(model_dict)

        detector._model = model

        return detector

    def train_fg(self, train_transactions: pd.DataFrame, params: Dict[str, Any] = None, test_mask: List[bool] = None):
        """
        Train FraudRGCN model on train_transactions.

        :param train_transactions: DataFrame with transaction to use for training.
        :param params: Optional. Overloads _default_params with these.
        :param test_mask: Optional. Array of booleans. where True value indicates a test transaction in train_transactions.
            Must be the same length as train_transactions.
        :return: self
        """

        self._params = dict(self._default_params)
        self._params.update({} if params is None else params)

        t1 = time.time()

        self._construct_graph(train_transactions,
                              self._params['target_col'],
                              self._params['node_cols'],
                              self._params['cat_cols'],
                              self._params['num_cols'])

        t2 = time.time()

        if self._params['num_gpus'] > 0:
            device = th.device('cuda:0')
        else:
            device = th.device('cpu')

        in_feats = self._train_g.nodes['target'].data['features'].shape[1]
        n_classes = 2

        ntype_dict = {n_type: self._train_g.number_of_nodes(n_type) for n_type in self._train_g.ntypes}

        model = HeteroRGCN(ntype_dict, self._train_g.etypes, in_feats, self._params['n_hidden'], n_classes, self._params['n_layers'], in_feats)
        model = model.to(device)

        print("Initialized Model")

        class_weights = [1. / self._params['class_weight'],
                         self._params['class_weight']]

        train_labels = train_transactions[self._params['label_col']].values

        if test_mask is None: ### when test_mask is None, model is trained in inductive mode
            test_mask = np.zeros_like(train_labels, dtype='bool')
        else: ### test_mask is passed to train model in transductive mode
            test_mask = np.asarray(test_mask)

        train_features = self._train_g.nodes['target'].data['features'].to(device)
        train_labels = th.from_numpy(train_labels).long().to(device)
        test_mask = th.from_numpy(test_mask).to(device)

        loss = th.nn.CrossEntropyLoss(weight=th.tensor(class_weights).float())
        optim = th.optim.Adam(model.parameters(), lr=self._params['lr'], weight_decay=self._params['weight_decay'])


        print("Starting Model training")
        model = train_fg(model, optim, loss, train_features, train_labels, self._train_g,
                         device, self._params['n_epochs'],
                         test_mask)

        print("Finished Model training")

        self._model = model

        t3 = time.time()

        self._timings['train: construct graph'].append(t2-t1)
        self._timings['train: fit model'].append(t3-t2)
        self._timings['train: total'].append(t3-t1)

        return self


    def _extend_graph(self, test_transactions, target_col, node_cols):

        features = np.nan_to_num(self._cat_transformer.transform(test_transactions), nan=0.)

        added_nodes = {}

        target_nodes, target_lookup, target_new_nodes, target_new_vals = encode_node_ids(test_transactions[target_col],
                                                                                         self._nodes_lookup['target'],
                                                                                         self._train_g.number_of_nodes('target'))

        target_new_nodes = set(target_new_nodes)

        target_nodes_to_add= [t for t in target_nodes if t in target_new_nodes]
        feature_sel= [True if t in target_new_nodes else False for t in target_nodes]

        new_features = np.compress(feature_sel, features, axis=0)
        new_features = normalize_test(th.from_numpy(new_features), self._train_mean, self._train_stdev)

        if len(target_new_nodes)> 0:
            self._train_g=dgl.add_nodes(self._train_g, len(target_new_nodes), ntype='target')
            self._train_g.nodes['target'].data['features'][-len(new_features):,:]=new_features
            added_nodes['target']=(list(target_new_nodes), target_new_vals)

        if len(target_nodes_to_add)>0:
            self._train_g = dgl.add_edges(self._train_g, target_nodes_to_add, target_nodes_to_add, etype=('target', 'self_relation', 'target'))

        for nc in node_cols.split(','):
            nodes, lookup, new_nodes, new_vals = encode_node_ids(test_transactions[nc],
                                                                 self._nodes_lookup[nc],
                                                                 self._train_g.number_of_nodes(nc))

            if len(new_nodes)> 0:
                self._train_g = dgl.add_nodes(self._train_g, len(new_nodes), ntype=nc)
                added_nodes[nc] = (new_nodes, new_vals)

            elist_u = []
            elist_v = []
            rlist_u = []
            rlist_v = []
            for s, t in zip(nodes, target_nodes):
                if t in target_new_nodes:
                    elist_u.append(t)
                    elist_v.append(s)
                    rlist_u.append(s)
                    rlist_v.append(t)

            if len(elist_u)>0:
                self._train_g = dgl.add_edges(self._train_g, elist_u, elist_v, etype=('target', f'target<>{nc}', nc))
                self._train_g = dgl.add_edges(self._train_g, rlist_u, rlist_v, etype=(nc, f'{nc}<>target', 'target'))

        return target_nodes, added_nodes

    def _construct_graph(self, train_transactions, target_col, node_cols, cat_cols, num_cols):
        """
        Helper method to construct graph object (dgl.heterograph).

        :param train_transactions: DataFrame with training transactions
        :param target_col: target column
        :param node_cols: comma-separated list of node columns
        :param cat_cols: comma-separated list of columns to use as categorical features
        :param num_cols: comma-separated list of columns to use as numerical features
        :return: None
        """

        self._cat_transformer= make_column_transformer(
            (
                FunctionTransformer(x_plus_log10),
                num_cols.split(',')
            ),
            (
                OneHotEncoder(handle_unknown='ignore', sparse=False),
                cat_cols.split(',')
            ),
            remainder='drop'
        )

        self._cat_transformer.fit(train_transactions)

        ### fill nan's with 0
        features = np.nan_to_num(self._cat_transformer.transform(train_transactions), nan=0.)

        ### create edge lists
        edgelists = {}

        self._nodes_lookup = {}
        self._nodes_lookup['target'] = {}

        ### transform target column to integer ids
        target_nodes, target_lookup, target_new_nodes, _ = encode_node_ids(train_transactions[target_col], self._nodes_lookup['target'], 0)
        self._nodes_lookup['target'] = target_lookup

        ### create self-relation edges
        edgelists[('target', 'self_relation', 'target')] = [(t, t) for t in target_nodes]

        for nc in node_cols.split(','):
            ### transform nodes of type nc to integer ids
            self._nodes_lookup[nc]={}
            nodes, lookup, new_nodes, _ = encode_node_ids(train_transactions[nc], self._nodes_lookup[nc], 0)
            self._nodes_lookup[nc] = lookup

            ### create bidirectional edges between target nodes and nodes of type nc
            elist = []
            rlist = []
            for s, t in zip(target_nodes, nodes):
                elist.append((s, t))
                rlist.append((t, s))

            edgelists[('target', f'target<>{nc}', nc)] = elist
            edgelists[(nc, f'{nc}<>target', 'target')] = rlist

        ### construct dgl.heterograph object from edge lists
        g = dgl.heterograph(edgelists)
        print(
            "Constructed heterograph with the following metagraph structure: Node types {}, Edge types{}".format(
                g.ntypes, g.canonical_etypes))
        print("Number of nodes of type target : {}".format(g.number_of_nodes('target')))

        g.nodes['target'].data['features'] = th.from_numpy(features.astype('float32'))

        self._train_mean, self._train_stdev, features = normalize_train(th.from_numpy(features.astype('float32')))

        g.nodes['target'].data['features'] = features

        self._train_g = g
