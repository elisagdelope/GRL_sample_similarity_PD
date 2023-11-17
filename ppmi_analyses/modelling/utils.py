import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix, from_networkx
import scipy.sparse as sp
import networkx as nx
from torchmetrics import MetricCollection, AUROC, Accuracy, Precision, Recall, Specificity, F1Score
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from PIL import Image
import wandb
from models import *
import shap
from torch_geometric.explain import Explainer, GNNExplainer

class MyPSN(InMemoryDataset):

    def __init__(self, root, X_file, graph_file, labels_cv_file, labels_test_file, transform=None, pre_transform=None,
                 pre_filter=None):  #
        self.X_file = X_file
        self.graph_file = graph_file
        self.labels_cv_file = labels_cv_file
        self.labels_test_file = labels_test_file
        super(MyPSN, self).__init__(root, transform, pre_transform, pre_filter=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return [self.X_file, self.graph_file, self.labels_cv_file, self.labels_test_file]

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        base_name = os.path.basename(self.X_file)
        root_name = os.path.splitext(base_name)[0].rsplit("_", 1)[0]
        processed_name = 'data_' + root_name + '.pt'
        return processed_name

    def download(self):
        """ Download to `self.raw_dir`.
            Not implemented here
        """
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):
        # load node attributes: gene expression (embeddings)
        X = pd.read_csv(self.raw_paths[0], index_col=0)  # index is patientID
        # map index to patientID in X -> {patientID: index}
        X_mapping = {index: i for i, index in enumerate(X.index.unique())}
        # load adjacency matrix & generate edge_index, edge_attr (weights)
        adj_matrix = pd.read_csv(self.raw_paths[1], index_col=0)
        edge_index, edge_attr = from_scipy_sparse_matrix(sp.csr_matrix(adj_matrix))
        if all(np.array(adj_matrix)[np.diag_indices_from(adj_matrix)] == 0):  # no self-loops
            try:
                assert edge_index.shape[1] == 2 * len(nx.from_pandas_adjacency(adj_matrix).edges)
            except AssertionError:
                print("edge_index has the wrong shape")
        else:  # self-loops
            try:
                assert edge_index.shape[1] == 2 * len(nx.from_pandas_adjacency(adj_matrix).edges) - adj_matrix.shape[0]
            except AssertionError:
                print("edge_index has the wrong shape")
        # load labels
        labels_cv = pd.read_csv(self.raw_paths[2], index_col=0)
        labels_test = pd.read_csv(self.raw_paths[3], index_col=0)
        labels = pd.concat([labels_cv, labels_test], axis=0)
        y = torch.tensor(np.array(labels)).squeeze(-1)
        # create train and test masks for data
        train_mask = torch.zeros(X.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(X.shape[0], dtype=torch.bool)
        train_mask[[X_mapping[x] for x in labels_cv.index]] = True
        test_mask[[X_mapping[x] for x in labels_test.index]] = True
        # build data object
        data = Data(edge_index=edge_index,
                    edge_attr=edge_attr,
                    x=torch.tensor(np.array(X)).type(torch.float),
                    y=y)
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        # data.num_nodes = G.number_of_nodes()
        # data.num_classes = 2
        # save processed data
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])


def train_epoch(device, model, data, criterion, optimizer, metric):
    """ Train step of model on training dataset (one epoch)
    """
    model.to(device)
    model.train()
    data.to(device)
    criterion.to(device)
    optimizer.zero_grad()  # Clear gradients
    # Perform a single forward pass #h, y_hat
    if "_uw" in str(model.__class__.__name__):  # for unweighted models
        y_hat = model(x=data.x, edge_index=data.edge_index)
    elif "GAT" in str(model.__class__.__name__):
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    else:
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32))
    loss = criterion(y_hat[data.train_mask], data.y[data.train_mask])  # Compute the loss
    loss.backward()  # Derive gradients
    optimizer.step()  # Update parameters based on gradients
    # track loss & embeddings
    tloss = loss.detach().cpu().numpy().item()
    # track performance
    y_hat = y_hat[:,1]  # get label
    batch_acc = metric(y_hat[data.train_mask].cpu(), data.y[data.train_mask].cpu())

    #emb_epoch = h.detach().cpu().numpy()  # tensor of all batches of the epoch
    train_acc = metric.compute()
    return tloss, train_acc #, emb_epoch

def evaluate_epoch(device, model, data, criterion, metric):
    """ Evaluate step of model on validation data
    """
    model.eval()
    model.to(device)
    data.to(device)
    criterion.to(device)
    # Perform a single forward pass # _, y_hat
    if "_uw" in str(model.__class__.__name__):  # for unweighted models
        y_hat = model(x=data.x, edge_index=data.edge_index)
    elif "GAT" in str(model.__class__.__name__):
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    else:
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32))
    if "val_mask" in data.items()._keys():
        vloss = criterion(y_hat[data.val_mask], data.y[data.val_mask])  # Compute the loss
        vloss = vloss.detach().cpu().numpy().item()
        y_hat = y_hat[:, 1]
        batch_vacc = metric(y_hat[data.val_mask].cpu(), data.y[data.val_mask].cpu())
    else:
        vloss = criterion(y_hat[data.test_mask],
                         data.y[data.test_mask])  # Compute the loss
        vloss = vloss.detach().cpu().numpy().item()
        y_hat = y_hat[:, 1]
        batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())

    val_acc = metric.compute()
    return vloss, val_acc

def test_epoch(device, model, data, metric):
    """ Evaluate step of model on test data
    """
    model.eval()
    data.to(device)
    # Perform a single forward pass # _, y_hat
    if "_uw" in str(model.__class__.__name__):  # for unweighted models
        y_hat = model(x=data.x, edge_index=data.edge_index)
    elif "GAT" in str(model.__class__.__name__):
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    else:
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32))
    y_hat = y_hat[:, 1]  # get label
    batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())
    test_acc = metric.compute()
    return test_acc

def training(device, model, optimizer, scheduler, criterion, data, n_epochs, fold):
    """ Full training process
    """
    losses = []
    #embeddings = []
    perf_metrics = {'Accuracy': [], 'AUC': [], 'Recall': [], 'Specificity': [], 'F1': []}
    train_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    val_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    for epoch in range(n_epochs):
        # train
        train_loss, train_perf = train_epoch(device, model, data, criterion, optimizer, train_metrics) #, epoch_embeddings
        # validation
        val_loss, val_perf = evaluate_epoch(device, model, data, criterion, val_metrics)
        # scheduler step
        scheduler.step(val_loss)
        # track losses & embeddings
        losses.append([train_loss, val_loss])
        #embeddings.append(epoch_embeddings)
        if "val_mask" in data.items()._keys():
            test_metrics = MetricCollection({
                'Accuracy': Accuracy(task="binary"),
                'AUC': AUROC(task="binary", num_classes=2),
                'Recall': Recall(task="binary", num_classes=2),
                'Specificity': Specificity(task="binary", num_classes=2),
                'F1': F1Score(task="binary", num_classes=2),
            })
            test_perf = test_epoch(device, model, data, test_metrics)
            for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(), test_perf[m].detach().numpy().item()])
            test_metrics.reset()
        else:
            for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item()])
        # log performance and loss in wandb
        wandb.log({f'val_loss-{fold}': val_loss,
                   f'train_loss-{fold}': train_loss,
                   f'val_Accuracy': val_perf["Accuracy"].detach().numpy().item(),
                   f'val_AUC-{fold}': val_perf["AUC"].detach().numpy().item(),
                   f'val_Recall-{fold}': val_perf["Recall"].detach().numpy().item(),
                   f'val_Specificity-{fold}': val_perf["Specificity"].detach().numpy().item(),
                   f'val_F1-{fold}': val_perf["F1"].detach().numpy().item(),
                   f'train_Accuracy-{fold}': train_perf["Accuracy"].detach().numpy().item(),
                   f'train_AUC-{fold}': train_perf["AUC"].detach().numpy().item(),
                   f'train_Recall-{fold}': train_perf["Recall"].detach().numpy().item(),
                   f'train_Specificity-{fold}': train_perf["Specificity"].detach().numpy().item(),
                   f'train_F1-{fold}': train_perf["F1"].detach().numpy().item(),
                   f'test_AUC-{fold}': test_perf["AUC"].detach().numpy().item(),
                   f'test_Accuracy-{fold}': test_perf["AUC"].detach().numpy().item(),
                   f'test_Recall-{fold}': test_perf["Recall"].detach().numpy().item(),
                   f'test_Specificity-{fold}': test_perf["Specificity"].detach().numpy().item(),
                   f'test_F1-{fold}': test_perf["F1"].detach().numpy().item()
                   }, step=epoch)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}",
                  f"Loss train {train_loss}",
                  f"Loss validation {val_loss}",
                  f"Acc train {train_perf}",
                  f"Acc validation {val_perf};")
        train_metrics.reset()
        val_metrics.reset()

        # identify best model based on max validation AUC
        if epoch < 1:
            best_loss = losses[epoch][1]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        else:
            if best_loss < losses[epoch][1]:
                continue
            else:
                best_loss = losses[epoch][1]
                best_model = copy.deepcopy(model)
                best_epoch = epoch
    return losses, perf_metrics, best_epoch, best_loss, best_model #, embeddings



def train_mlp(device, model, optimizer, criterion, data, metric):
    model.train()
    data.to(device)
    optimizer.zero_grad()  # Clear gradients.
    y_hat = model(data.x)  # Perform a single forward pass.
    loss = criterion(y_hat[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    # track loss & embeddings
    epoch_loss = loss.detach().cpu().numpy().item()
    #y_hat = torch.max(y_hat, 1)[1]  # get label
    y_hat = y_hat[:,1]
    batch_perf = metric(y_hat[data.train_mask].cpu(), data.y[data.train_mask].cpu())
    train_perf = metric.compute()
    return epoch_loss, train_perf

def evaluate_mlp(device, model, criterion, data, metric):
    model.eval()
    data.to(device)
    y_hat = model(data.x) #_, y_hat
    if "val_mask" in data.items()._keys():
        vloss = criterion(y_hat[data.val_mask], data.y[data.val_mask])  # Compute the loss
        vloss = vloss.detach().cpu().numpy().item()
        y_hat = y_hat[:, 1]
        batch_vacc = metric(y_hat[data.val_mask].cpu(), data.y[data.val_mask].cpu())
    else:
        vloss = criterion(y_hat[data.test_mask],
                         data.y[data.test_mask])  # Compute the loss
        vloss = vloss.detach().cpu().numpy().item()
        y_hat = y_hat[:, 1]
        batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())

    val_perf = metric.compute()
    return vloss, val_perf

def test_mlp(device, model, data, metric):
    model.eval()
    data.to(device)
    y_hat = model(data.x) # _, y_hat
    y_hat = y_hat[:,1]
    batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())
    test_perf = metric.compute()
    return test_perf

def training_mlp(device, model, optimizer, scheduler, criterion, data, n_epochs, fold):
    losses = []
    perf_metrics = {'Accuracy': [], 'AUC': [], 'Recall': [], 'Specificity': [], 'F1': []}
    train_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    val_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    for epoch in range(n_epochs):
        # train
        train_loss, train_perf = train_mlp(device, model, optimizer, criterion, data, train_metrics)
        # validation
        val_loss, val_perf = evaluate_mlp(device, model, criterion, data, val_metrics)
        # scheduler step
        scheduler.step(val_loss)
        # track losses & embeddings
        losses.append([train_loss, val_loss])
        if "val_mask" in data.items()._keys():
            test_metrics = MetricCollection({
                'Accuracy': Accuracy(task="binary"),
                'AUC': AUROC(task="binary", num_classes=2),
                'Recall': Recall(task="binary", num_classes=2),
                'Specificity': Specificity(task="binary", num_classes=2),
                'F1': F1Score(task="binary", num_classes=2),
            })
            test_perf = test_mlp(device, model, data, test_metrics)
            for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(),
                                        test_perf[m].detach().numpy().item()])
            test_metrics.reset()
        else:
            for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item()])
        # log performance and loss in wandb
        wandb.log({f'val_loss-{fold}': val_loss,
                   f'train_loss-{fold}': val_loss,
                   f'val_Accuracy': val_perf["Accuracy"].detach().numpy().item(),
                   f'val_AUC-{fold}': val_perf["AUC"].detach().numpy().item(),
                   f'val_Recall-{fold}': val_perf["Recall"].detach().numpy().item(),
                   f'val_Specificity-{fold}': val_perf["Specificity"].detach().numpy().item(),
                   f'val_F1-{fold}': val_perf["F1"].detach().numpy().item(),
                   f'train_Accuracy-{fold}': train_perf["Accuracy"].detach().numpy().item(),
                   f'train_AUC-{fold}': train_perf["AUC"].detach().numpy().item(),
                   f'train_Recall-{fold}': train_perf["Recall"].detach().numpy().item(),
                   f'train_Specificity-{fold}': train_perf["Specificity"].detach().numpy().item(),
                   f'train_F1-{fold}': train_perf["F1"].detach().numpy().item(),
                   f'test_AUC-{fold}': test_perf["AUC"].detach().numpy().item(),
                   f'test_Accuracy-{fold}': test_perf["AUC"].detach().numpy().item(),
                   f'test_Recall-{fold}': test_perf["Recall"].detach().numpy().item(),
                   f'test_Specificity-{fold}': test_perf["Specificity"].detach().numpy().item(),
                   f'test_F1-{fold}': test_perf["F1"].detach().numpy().item()
                   }, step=epoch)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}",
                  f"Loss train {train_loss}",
                  f"Loss validation {val_loss}",
                  f"Acc train {train_perf}",
                  f"Acc validation {val_perf};")
        train_metrics.reset()
        val_metrics.reset()
        # identify best model based on max validation AUC
        if epoch < 1:
            best_loss = losses[epoch][1]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        else:
            if best_loss < losses[epoch][1]:
                continue
            else:
                best_loss = losses[epoch][1]
                best_model = copy.deepcopy(model)
                best_epoch = epoch
    return losses, perf_metrics, best_epoch, best_loss, best_model

def embeddings_2pca(embeddings):
    """ Generates 3-dimensional pca from d-dimensional embeddings.
        Returns a pandas dataframe with the 3-d pc.
    """
    pca = PCA(n_components=3, random_state=42)
    pca_result = pca.fit_transform(embeddings)
    pca_df = pd.DataFrame()
    pca_df['pca-v1'] = pca_result[:, 0]
    pca_df['pca-v2'] = pca_result[:, 1]
    pca_df['pca-v3'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    return pca_df


def k_fold(x, y, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    test_mask, train_mask = [], []
    mask_array = torch.zeros(x.shape[0], dtype=torch.bool)
    for _, idx in skf.split(torch.zeros(x.shape[0]), y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))
        mask_array = torch.zeros(x.shape[0], dtype=torch.bool)
        mask_array[test_indices[-1]] = True
        test_mask.append(mask_array)

    val_indices = [test_indices[i - 1] for i in range(folds)]
    val_mask = [test_mask[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask_indices = torch.ones(x.shape[0], dtype=torch.bool)
        train_mask_indices[test_indices[i]] = 0
        train_mask_indices[val_indices[i]] = 0
        train_indices.append(train_mask_indices.nonzero(as_tuple=False).view(-1))
        mask_array = torch.zeros(x.shape[0], dtype=torch.bool)
        mask_array[train_indices[-1]] = True
        train_mask.append(mask_array)

    #return train_indices, test_indices, val_indices
    return train_mask, test_mask, val_mask


def cv_training(folds, n_epochs, model, lr, wd, dataset, scheduler_factor=0.2, scheduler_threshold=0.0001, scheduler_patience=15):

    fold_v_performance = {'Fold': [],'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [], 'N_epoch': [], 'Loss': []}
    fold_test_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [], 'N_epoch': []}
    for fold, (train_msk, test_msk, val_msk) in enumerate(zip(*k_fold(dataset, folds))):
        # create cv masks
        dataset.data["train_mask"] = train_msk
        dataset.data["val_mask"] = val_msk
        dataset.data["test_mask"] = test_msk
        data = dataset[0]

        # compute class weights for loss function
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(data.y[data.train_mask]),
                                                          y=data.y[data.train_mask].numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=scheduler_factor,
                                                   threshold=scheduler_threshold, patience=scheduler_patience,
                                                   verbose=True)

        if "MLP" in model._get_name():
            # training for non graph methods
            losses, performance, best_epoch, best_loss, best_model = training_mlp(device, model, optimizer, scheduler, criterion, data, n_epochs)
        else:
            # training for graph methods
            losses, performance, best_epoch, best_loss, best_model = training(device, model, optimizer, scheduler, criterion, data, n_epochs) # , embeddings

        # calculate best epoch
    #    v_loss = [epoch_loss[1] for epoch_loss in losses]
    #    v_loss = torch.tensor(v_loss)
    #    best_epoch1 = v_loss.argmin()

        # collect cv performance
        fold_v_performance["Fold"].append(fold)
        fold_test_performance["Fold"].append(fold)
        fold_v_performance["N_epoch"].append(best_epoch)
        fold_test_performance["N_epoch"].append(best_epoch)
        for m in performance.keys():
            fold_v_performance[m].append(performance[m][best_epoch][1])
            fold_test_performance[m].append(performance[m][best_epoch][2])
        fold_v_performance["Loss"].append(losses[best_epoch][1])

        # print performance
        v_perf = [epoch_perf[1] for epoch_perf in performance["AUC"]]
        v_perf = torch.tensor(v_perf)
        best_epoch2 = v_perf.argmax()
        eval_info = {
            'fold': fold,
            'epoch': best_epoch,  # number of best epoch
            'val_loss': losses[best_epoch][1],  # val_loss at best epoch
            'val_auc': performance["AUC"][best_epoch][1],  # val_perf at best epoch
            'test_auc': performance["AUC"][best_epoch][2]  # test_AUC at best epoch
        }
        print(best_epoch, performance["AUC"][best_epoch][1], performance["AUC"][best_epoch][2],
              best_epoch2, performance["F1"][best_epoch2][1], performance["AUC"][best_epoch2][1],
              performance["AUC"][best_epoch2][2])
        print(eval_info)

        # reset parameters
        print('*resetting model parameters*')
        for name, module in model.named_children():
            module.reset_parameters()

    df_fold_v = pd.DataFrame(fold_v_performance)
    df_fold_v.set_index("Fold", inplace=True)
    df_fold_test = pd.DataFrame(fold_test_performance)
    df_fold_test.set_index("Fold", inplace=True)
    return(df_fold_v, df_fold_test)


def build_sparse_mask(dense_mask, k):
    sparse_mask = torch.tensor(np.array(dense_mask)).nonzero()
    in_features = pw_mask.shape[0]
    out_features = pw_mask.shape[1] * k

    k_connections = torch.tensor([], dtype=int)
    for i in range(1, pw_K):
        temp = copy.deepcopy(sparse_mask)
        temp[:, 1] = temp[:, 1] + (pw_mask.shape[1] * i)
        k_connections = torch.cat((k_connections, temp), dim=0)
    sparse_mask = torch.cat((sparse_mask, k_connections), dim=0)

    return sparse_mask, in_features, out_features



def get_results(results_dict):
    raw_results = np.stack(list(results_dict.values()))
    results = pd.DataFrame(columns=list(results_dict.keys()))
    name_row = list()
    for i, metric in enumerate(list(list(results_dict.values())[0].columns)):
        df_mean = pd.DataFrame(
            data=np.mean(raw_results[:, :, i], axis=1)).T
        df_mean.columns = results.columns
        results = results.append(df_mean)
        name_row.append("%s" % 'mean' + "_%s" % metric)

        df_std = pd.DataFrame(
            data=np.std(raw_results[:, :, i], axis=1)).T
        df_std.columns = results.columns
        results = results.append(df_std)
        name_row.append("%s" % 'std' + "_%s" % metric)

        df_max = pd.DataFrame(
            data=np.max(raw_results[:, :, i], axis=1)).T
        df_max.columns = results.columns
        results = results.append(df_max)
        name_row.append("%s" % 'max' + "_%s" % metric)

        df_min = pd.DataFrame(
            data=np.min(raw_results[:, :, i], axis=1)).T
        df_min.columns = results.columns
        results = results.append(df_min)
        name_row.append("%s" % 'min' + "_%s" % metric)

    for j in range(raw_results.shape[1]):
        df_temp = pd.DataFrame(
            data=raw_results[:, j, 0]) # AUC is in column index 0
        df_temp = df_temp.T
        df_temp.columns = results.columns
        results = results.append(df_temp)
        name_row.append("Split %i" % (j + 1))

    results.index = name_row
    return results

def is_symmetric(matrix: np.ndarray) -> bool:
    return np.array_equal(matrix, matrix.T)


def find_unconnected(adj_matrix):
    # Create a list of all the nodes
    nodes = list(range(len(adj_matrix)))
    # Use a lambda function to check if a row in the adjacency matrix has any non-zero entries
    has_connections = lambda row: any(x != 0 for x in row)

    # Use the filter function to find the nodes with no connections
    unconnected_nodes = list(filter(lambda i: not has_connections(adj_matrix[i]), nodes))
    return unconnected_nodes


def add_edges(sim_matrix, adj_matrix):
    unconnected = find_unconnected(np.array(adj_matrix))
    print("There are %d unconnected nodes for which to add edges" % len(unconnected))

    # remove self-loops from similarity matrix
    a = np.array(sim_matrix)
    a[np.diag_indices_from(a)] = 0.
    sim_matrix = pd.DataFrame(a, index=sim_matrix.index, columns=sim_matrix.columns)

    # column name of strongest edge for unconnected nodes
    idx_strongesc = sim_matrix.iloc[unconnected, :].idxmax(axis=1)
    val_strongesc = sim_matrix.iloc[unconnected, :].max(axis=1)
    strongest_edges = sim_matrix.iloc[unconnected, :].max(axis=1)

    # min similarity of the strongest edge for unconnected nodes
    print("The weakest edge that was added for unconnected nodes has a similarity of %f" % min(strongest_edges))

    # Substitute the 0 value in adjacency matrix with that of the strongest edge in unconnected nodes
    for i, col_name in idx_strongesc.items():
        adj_matrix.loc[i, col_name] = val_strongesc[i]
        adj_matrix.loc[col_name, i] = val_strongesc[i]  # also the other side

    return adj_matrix


def connect_components_with_strongest_links(adj_df, sim_matrix):
    """
    Connect all connected components in a graph defined by its adjacency matrix adj_df by adding the minimum
    number of links, using the strongest links (based on a similarity matrix sim_matrix) that connect all components together.
    Returns the adjacency matrix of a connected graph.
    """
    # Create a graph from the adjacency matrix
    G = nx.from_pandas_adjacency(adj_df)
    components = list(nx.connected_components(G))
    num_components = len(components)
    if num_components > 1:
        strongest_links=[]
        for i in range(num_components):
            for j in range(i + 1, num_components):
                component1 = components[i]
                component2 = components[j]
                # Find the strongest link between component1 and component2
                strongest_link = sim_matrix.loc[component1, component2].unstack().idxmax()
                strongest_node1, strongest_node2 = strongest_link[0], strongest_link[1]
                max_link_strength = sim_matrix.loc[strongest_node1, strongest_node2]
                # Add the strongest link to the list of strongest links
                strongest_links.append((strongest_node1, strongest_node2, max_link_strength))
        # Sort the strongest links based on link strength in descending order
        strongest_links.sort(key=lambda x: x[2], reverse=True)
        # Connect all components using the strongest links
        link_count=0
        strength_track=[]
        for link in strongest_links:
            node1, node2, strength = link
            # Check if node1 and node2 are still in separate components
            component1 = nx.node_connected_component(G, node1)
            component2 = nx.node_connected_component(G, node2)
            if component1 != component2:
                # Update the (symmetric) adjacency matrix with the strongest link
                adj_df.loc[node1, node2] = strength
                adj_df.loc[node2, node1] = strength
                G.add_edge(node1, node2)  # Add edge to the graph
                # Update the connected components list
                components = list(nx.connected_components(G))
                link_count +=1
                strength_track.append(strength)
                # Check if all components are connected
                if len(components) == 1:
                    break
        print(f'{link_count} edges were added to obtain a connected graph.')
        print(f'The weakest edge that was added has a similarity of {min(strength_track)}.')
        # explore the new graph
        explore_graph(adj_df)
    else:
        print("The graph is connected; no need to add edges")
    return adj_df




def calculate_homophily_ratio(G, attr_name):
    """
    Calculate the homophily ratio for a given attribute in a graph.
    The homophily ratio is the ratio of connected pairs of nodes with the same labels with respect to
    connected pairs of nodes with different labels
    Args:
    G (nx.Graph): The graph.
    attr_name (str): The name of the node attribute to calculate the homophily ratio for.

    Returns:
    float: The homophily ratio for the given attribute.
    """
    # Get the set of unique values for the attribute
    attr_values = set(nx.get_node_attributes(G, attr_name).values())

    # Initialize variables to store the number of connected pairs of nodes with the same and different attributes
    same_attr_connected_pairs = 0
    different_attr_connected_pairs = 0

    # Iterate over the unique values of the attribute
    for attr_value in attr_values:
        # Get the set of nodes with the given attribute value
        attr_value_nodes = [node for node, attr in nx.get_node_attributes(G, attr_name).items() if attr == attr_value]

        # Iterate over all pairs of nodes with the given attribute value
        for i, node1 in enumerate(attr_value_nodes):
            for j, node2 in enumerate(attr_value_nodes):
                # Skip self-loops and edges already evaluated
                if i >= j:
                    continue
                # Check if the nodes are connected
                if node1 in G[node2] or node2 in G[node1]:
                    # Increment the number of connected pairs with the same attribute
                    same_attr_connected_pairs += 1

        # Get the set of nodes with different attribute values
        different_attr_nodes = [node for node, attr in nx.get_node_attributes(G, attr_name).items() if
                                attr != attr_value]

        # Iterate over all pairs of nodes with different attribute values
        for i, node1 in enumerate(attr_value_nodes):
            for j, node2 in enumerate(different_attr_nodes):
                # Skip self-loops and edges already evaluated
                if i >= j:
                    continue
                # Check if the nodes are connected
                if node1 in G[node2] or node2 in G[node1]:
                    # Increment the number of connected pairs with different attributes
                    different_attr_connected_pairs += 1
    # Calculate and return the homophily ratio
    return same_attr_connected_pairs / different_attr_connected_pairs


def display_graph(fold, adj_df, labels_dict=None, save_fig=False, path="./", name_file="graph.png", plot_title=None):
    """Draw the graph given an adjacency matrix"""
    fig = plt.figure(figsize=(12, 12))
    G = nx.from_pandas_adjacency(adj_df)
    weights = nx.get_edge_attributes(G, 'weight').values()
    pos = nx.spring_layout(G)
    if labels_dict is None:
        nx.draw(G, pos=pos, with_labels=False,
                cmap=plt.get_cmap("viridis"), node_color="blue", node_size=80,
                width=list(weights), ax=fig.add_subplot(111))
    else:
        nx.set_node_attributes(G, labels_dict, "label")
        # Get the values of the labels
        l = list(nx.get_node_attributes(G, "label").values())
        color_map = {"dodgerblue": 0, "red": 1}  # blue for controls, red for disease
        color_map = {v: k for k, v in color_map.items()}
        colors = [color_map[cat] for cat in l]
        # Draw the graph
        nx.draw(G, pos=pos, with_labels=False,
                cmap=plt.get_cmap("viridis"), node_color=colors, node_size=80,
                width=list(weights), ax=fig.add_subplot(111))
    plt.title(plot_title, fontsize=24)
    plt.tight_layout()
    if save_fig:
        fig.savefig(path + name_file)
    # Log the image to wandb: Convert the graph image to a PIL Image
    image = Image.frombytes('RGB', fig.canvas.get_width_height(),
                            fig.canvas.tostring_rgb())
    wandb.log({f'graph-{fold}': wandb.Image(image), "caption": "Graph Visualization"})
    plt.close(fig)

def explore_graph(adj_df):
    G = nx.from_pandas_adjacency(adj_df)
    print("N nodes in G:", len(G.nodes))
    print("N edges in G:", len(G.edges))
    print("N non-zero elements in adjacency matrix:", np.count_nonzero(np.array(adj_df)))  # make sure it should be 2* number of edges or [2*number_edges - number_nodes] if self-loops exist
    print("G is directed:", nx.is_directed(G))
    print("G is connected:", nx.is_connected(G))
    print("G is weighted:", nx.is_weighted(G))
    # connected components
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    print(f"There are {len(Gcc)} components in G")
    print("Size of largest connected component:", len(Gcc[0]))
    # degree distribution
    degrees = [d for n, d in G.degree()]
    print("Degree distribution:")
    print(pd.DataFrame(pd.Series(degrees).describe()).transpose().round(2))

def similarity_network(s_threshold, X_df):
    """
    Build a similarity network from a given DataFrame using a threshold for similarity values.
    The network is constructed by calculating pairwise cosine similarity between data points, and then thresholding
    the similarity values to create an adjacency matrix. The adjacency matrix is then checked for symmetry and
    connectivity, and edges are added to ensure a connected graph.

    Parameters:
        - s_threshold (float): Threshold value for similarity. Similarity values below this threshold are set to 0.
        - X_df (pandas DataFrame): DataFrame containing data points as rows and features as columns.

    Returns:
        - adj_df (pandas DataFrame): Adjacency matrix representing the similarity network.
    """
    # Calculate pairwise cosine distance between data points
    dist = pd.DataFrame(
        squareform(pdist(X_df, metric='cosine')),
        columns=X_df.index,
        index=X_df.index
    )
    # Calculate similarity from distance
    sim = 1 - dist
    adj_df = sim[sim > s_threshold]
    adj_df = adj_df.fillna(0)
    a = np.array(adj_df)
    a[np.diag_indices_from(a)] = 0.  # let's temporarily remove self-loops
    adj_df = pd.DataFrame(a, index=adj_df.index, columns=adj_df.columns)
    print("avg adj", np.mean(np.array(adj_df)))
    if not is_symmetric(np.array(adj_df)):
        raise ValueError('Adjacency matrix is not symmetric')
    # check for unconnected nodes and connect them to their most similar peer
    if (len(find_unconnected(np.array(adj_df))) > 0):
        print(f'Number of unconnected nodes: {len(find_unconnected(np.array(adj_df)))} out of {adj_df.shape[0]} nodes in G')
        adj_df = add_edges(sim, adj_df)
        print("avg adj", np.mean(np.array(adj_df)))
        if not is_symmetric(np.array(adj_df)):
            raise ValueError('Adjacency matrix is not symmetric')
    # explore the graph
    G = nx.from_pandas_adjacency(adj_df)
    explore_graph(adj_df)
    # check for unconnected components and connect them with the minimum number of the strongest links
    adj_df = connect_components_with_strongest_links(adj_df, sim)
    return adj_df


def create_pyg_data(adj_df, X_df, y, train_msk, val_msk, test_msk):
    edge_index, edge_attr = from_scipy_sparse_matrix(sp.csr_matrix(adj_df))
    data = Data(edge_index=edge_index,
                edge_attr=edge_attr,
                x=torch.tensor(X_df.values).type(torch.float),
                y=torch.tensor(y))
    data["train_mask"] = train_msk
    data["val_mask"] = val_msk
    data["test_mask"] = test_msk
    # Gather and show some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    unique, counts = np.unique(data.y, return_counts=True)
    print("Classes:", unique)
    print("Counts:", counts)
    return data


def update_overall_metrics(fold, fold_best_epoch, homophily_index, feat_names, relevant_features, fold_performance, fold_losses, dict_val_metrics, dict_test_metrics, features_track):
    dict_val_metrics["Fold"].append(fold)
    dict_val_metrics["N_epoch"].append(fold_best_epoch)
    dict_val_metrics["N_features"].append(len(feat_names))
    dict_val_metrics["homophily_index"].append(homophily_index)
    dict_test_metrics["Fold"].append(fold)
    dict_test_metrics["N_epoch"].append(fold_best_epoch)
    features_track["Fold"].append(fold)
    features_track["N_selected_features"].append(len(feat_names))
    features_track["Selected_Features"].append(feat_names)
    features_track["Relevant_Features"].append(relevant_features)
    for m in fold_performance.keys():
        dict_val_metrics[m].append(fold_performance[m][fold_best_epoch][1])
        dict_test_metrics[m].append(fold_performance[m][fold_best_epoch][2])
    dict_val_metrics["Loss"].append(fold_losses[fold_best_epoch][1])
    return (dict_val_metrics, dict_test_metrics, features_track)

def cv_metrics_to_wandb(dict_val_metrics, dict_test_metrics):
    for key in dict_val_metrics.keys():
        val_values = dict_val_metrics[key]
        mean_val = np.mean(val_values)
        std_val = np.std(val_values)
        wandb.run.summary[f"mean_val_{key}"] = mean_val
        wandb.run.summary[f"std_val_{key}"] = std_val
        wandb.run.summary[f"values_val_{key}"] = np.array(val_values)
        wandb.log({f"mean_val_{key}": mean_val, f"std_val_{key}": std_val}, commit=False)
        if key in dict_test_metrics.keys():
            test_values = dict_test_metrics[key]
            mean_test = np.mean(test_values)
            std_test = np.std(test_values)
            wandb.run.summary[f"mean_test_{key}"] = mean_test
            wandb.run.summary[f"std_test_{key}"] = std_test
            wandb.run.summary[f"values_test_{key}"] = np.array(test_values)
            wandb.log({f"mean_test_{key}": mean_test, f"std_test_{key}": std_test}, commit=False)

def generate_model(model_name, config, n_features):
    models_dict = {"MLP2": MLP2(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout),
                   "GCNN_uw": GCNN_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout),
                   "GCNN": GCNN(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout),
                   "Cheb_GCNN_uw": Cheb_GCNN_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.K_cheby, config.ll_out_units, config.dropout),
                   "Cheb_GCNN": Cheb_GCNN(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.K_cheby, config.ll_out_units, config.dropout),
                   "GAT": GAT(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout),
                   "GAT_uw": GAT_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads,
                              config.ll_out_units, config.dropout)
    }
    model = models_dict[model_name]
    print(model)
    return model


class GraphMasker:
    def __init__(self, data):
        self.data = data
        self.x = data.x
        self.edge_index = data.edge_index
        self.shape = data.x.shape

    def __call__(self, mask):
        self.x = torch.where(mask[:, None].bool(), torch.zeros_like(self.x), self.x)
        self.data.x = self.x
        self.data.edge_index = self.edge_index
        return self.data.x  # Return the x attribute as a Tensor


def plot_shap_values(device, model, data, save_fig=False, name_file=None, path=None, names_list=None, plot_title=None):
    # Convert data to PyTorch Geometric format
    data = data.to(device)
    edge_index = data.edge_index
    # Initialize NodeConductance explainer
    explainer = NodeConductance(model)
    # Calculate Shapley values for each node
    shap_values = []
    for i in range(data.num_nodes):
        node_feat_mask = torch.zeros(data.num_nodes).to(device)
        node_feat_mask[i] = 1
        attr = explainer.attribute(data.x, additional_forward_args=(edge_index, node_feat_mask), target=None)
        shap_values.append(attr.sum(dim=0).detach().cpu().numpy())
    shap_values = np.array(shap_values)
    if not names_list:
        names_list = list(X.columns)
    shap_fig = shap.summary_plot(shap_values=shap_values.values, features=X, feature_names=names_list, show=False)
#    shap.plots.beeswarm(shap_values, show=False)
    fig, ax = plt.gcf(), plt.gca()
    labels = ax.get_yticklabels()
    fig.axes[-1].set_aspect(150)  # set the color bar
    fig.axes[-1].set_box_aspect(150)  # set the color bar
    fig.set_size_inches(15, 10)  # set figure size
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xticklabels(np.round(ax.get_xticks(), 2), rotation=15, fontsize=8)
    plt.title(plot_title)
    fig.tight_layout()
    if save_fig:
        fig.savefig(path + name_file)
 #   fig.show()
    # Log the image to wandb: Convert the graph image to a PIL Image
    image = Image.frombytes('RGB', fig.canvas.get_width_height(),
                            fig.canvas.tostring_rgb())
    wandb.log({f'shap-{fold}': wandb.Image(image), "caption": "Shap values analysis"})
    plt.close(fig)

def calculate_feature_importance(model, data, names_list=None, save_fig=False, name_file='feature_importance', path=None, n=20):
    """
    Explainability at node and node features' level as per GNN-Explainer model from the “GNNExplainer: Generating
    Explanations for Graph Neural Networks” paper for identifying compact subgraph structures and node features
    that play a crucial role in the predictions made by a GNN
    :param model:
    :param data:
    :param names_list:
    :param save_fig:
    :param name_file:
    :param path:
    :param n:
    :return:
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    if "MLP" in str(model.__class__.__name__):  # for unweighted models
        explanation = explainer(x=data.x)
    elif "_uw" in str(model.__class__.__name__):  # for unweighted models
        explanation = explainer(x=data.x, edge_index=data.edge_index)
    elif "GAT" in str(model.__class__.__name__):
        explanation = explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    else:
        explanation = explainer(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32))
    print(f'Generated explanations in {explanation.available_explanations}')
    if save_fig:
        feat_importance = explanation.visualize_feature_importance(str(path) + name_file + ".png",
                                                                   top_k=n, feat_labels=names_list)
        print(f"Feature importance plot has been saved to '{path}'")
        node_importance = explanation.visualize_graph(path + name_file + "_subgraph.pdf")
        print(f"Subgraph visualization plot has been saved to '{path}'")
    else:
        feat_importance = explanation.visualize_feature_importance(path=None,
                                                                   top_k=n, feat_labels=names_list)
        node_importance = explanation.visualize_graph(path=None)
    return feat_importance, node_importance




def feature_importances_shap_values(model, data, X, names_list=None, n=20):
    """
    Extracts the top n relevant features based on SHAP values in an ordered way

    Parameters
    ----------
    model : instance of BaseEstimator from scikit-learn.
            Model for which shap values have to be computed.
    X : array, shape (n_samples, n_features).
        Training data.
    names_list : list
                 Names of the features (length # features) to appear in the plot.
    n : number of features to retrieve
    """
    # generate shap values
    masker = shap.maskers.Independent(data)
    explainer = shap.DeepExplainer(model, masker=masker)
    shap_values = explainer(data)
    if np.ndim(shap_values) == 3:
        shap_values = shap_values[:, :, 1]
    if not names_list:
        names_list = list(X.columns)
    shap_df = pd.DataFrame(shap_values.values, columns=names_list)
    vals = np.abs(shap_df).mean(0)
    shap_importance = pd.DataFrame(list(zip(names_list, vals)),
                                   columns=['feature', 'shap_value'])
    shap_importance.sort_values(by=['shap_value'],
                                ascending=False,
                                inplace=True)
    shap_importance = shap_importance.iloc[0:n, ]
    return shap_importance
