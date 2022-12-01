
import numpy as np
import time
import torch as th
from sklearn.metrics import confusion_matrix

def x_plus_log10(x):
    return np.log10(x + 1e-9)

def encode_node_ids(vals, lookup, new_node_id=0):
    ret=[]
    new_nodes=[]
    new_vals=[]
    for v in vals:
        if v not in lookup:
            lookup[v] = new_node_id
            new_nodes.append(new_node_id)
            new_vals.append(v)
            new_node_id += 1

        ret.append(lookup[v])

    return ret, lookup, new_nodes, new_vals

def normalize_test(feature_matrix, mean, stdev):
    return (feature_matrix - mean) / stdev

def normalize_train(feature_matrix):
    mean = th.mean(feature_matrix, dim=0)
    stdev = th.sqrt(th.sum((feature_matrix - mean)**2, dim=0)/feature_matrix.shape[0])
    stdev = stdev.numpy()
    stdev[np.isclose(stdev, 0.0)]=1.
    stdev = th.from_numpy(stdev)
    return mean, stdev, (feature_matrix - mean) / stdev

def train_fg(model, optim, loss, features, labels, train_g, device, n_epochs, test_mask):
    """
    A full graph verison of RGCN training
    """

    train_mask = th.logical_not(test_mask)
    train_idx =  th.nonzero(train_mask, as_tuple=True)[0]

    duration = []
    for epoch in range(n_epochs):
        tic = time.time()
        loss_val = 0.

        pred = model(train_g, features.to(device))
        l = loss(th.index_select(pred, 0, train_idx), 
                 th.index_select(labels, 0, train_idx))

        optim.zero_grad()
        l.backward()
        optim.step()

        loss_val += l

        duration.append(time.time() - tic)
        metric = evaluate(model, train_g, features, labels, test_mask, device)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | f1 {:.4f} ".format(
                epoch, np.mean(duration), loss_val, metric))

    return model


def get_f1_score(y_true, y_pred):
    """
    Only works for binary case.
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]

    :param y_true: A list of labels in 0 or 1: 1 * N
    :param y_pred: A list of labels in 0 or 1: 1 * N
    :return:
    """

    cf_m = confusion_matrix(y_true, y_pred)

    if cf_m.shape[0]==1 and cf_m.shape[1]==1: # edge case when all labels are negative
        return 1.,1.,1.

    precision = cf_m[1,1] / (cf_m[1,1] + cf_m[0,1] + 10e-5)
    recall = cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])
    f1 = 2 * (precision * recall) / (precision + recall + 10e-5)

    return precision, recall, f1


def evaluate(model, g, features, labels, test_mask, device):
    "Compute the F1 value in a binary classification case"

    preds = model(g, features.to(device))
    preds = th.argmax(preds, dim=1).numpy()
    train_mask = np.logical_not(test_mask.numpy().astype('bool'))
    train_labels = labels.numpy()
    train_labels = np.compress(train_mask, train_labels)
    train_preds= np.compress(train_mask, preds)

    precision, recall, f1 = get_f1_score(train_labels, train_preds)

    return f1
