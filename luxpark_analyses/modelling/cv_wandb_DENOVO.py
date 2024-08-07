import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from matplotlib import pyplot as plt
from utils import *
from models import *
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import networkx as nx
import wandb
import yaml
from pathlib import Path
from torch_geometric.utils import homophily
import argparse
import shap
from datetime import date
from imblearn.under_sampling import RandomUnderSampler
import random 

#---
#sweep_config = {'project': 'my-cv-test5',
#                'method': 'grid',
#                'program': 'cv_test.py',
#                'metric': {'name': 'val_loss', 'goal': 'minimize'},
#                'parameters': {'C_lasso': {'values': [0.1, 0.05]},
#                               'n_folds': {'value': 5},
#                               'S_threshold': {'values': [0.6]},
#                               'model_name': {'value': "Cheb_GCNN_uw"},
#                               'n_epochs': {'value': 50},
#                               'lr': {'value': 0.03},
#                               'lrscheduler_factor': {'value': 0.9},
#                               'dropout': {'value': 0.5},
#                               'weight_decay': {'value': 0.01},
#                               'cl1_hidden_units': {'value': 8},
#                               'cl2_hidden_units': {'value': 8},
#                               'll_out_units': {'value': 2},
#                               'K_cheby': {'value':2},
#                               'heads': {'value':3}}}
#wandb.login()
#sweep_id = wandb.sweep(sweep=sweep_config, project='my-cv-test5') # set up environment variable??

if __name__ == '__main__':
    #os.environ["WANDB_API_KEY"] = "834f701bb72a96a9988d59e3d9d710f5f9a4b9ef"
    # set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='sweep_config', type=str) # , default='configs/default.yaml'
    args, unknown = parser.parse_known_args()
    # set up wandb
    sweep_run = wandb.init(config=args.sweep_config)
    myconfig = wandb.config
    print('Config file from wandb:', myconfig)
    if torch.cuda.is_available():
        print('cuda available')
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        device = torch.device('cuda')
    else:
        print('cuda not available')
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        device = torch.device('cpu')
    undersampling = True
    # I/O
    OUT_DIR = "../results/wandb/"
    datestamp = date.today().strftime('%Y%m%d')
    metab_file = "../data/DENOVO_data_metab_4ML_DIAGNOSIS.tsv"
    metab = pd.read_csv(metab_file, sep='\t', index_col=0)
    print("Dimensions of dataset:", metab.shape)
    # load X and y
    y = metab["DIAGNOSIS"]
    metab.drop(["DIAGNOSIS"], axis=1, inplace=True)
    y = y[y.index.isin([str(i) for i in metab.index.to_list()])]

    if undersampling:
        rus = RandomUnderSampler(random_state=42)
        X, y = rus.fit_resample(metab, y)
        X.index = metab.index[rus.sample_indices_]
        y.index = metab.index[rus.sample_indices_]
        labels_dict = y.to_dict()
        labels_dict = {k: int(v) for k, v in labels_dict.items()}
        features_name = metab.columns
        y = np.array(y)
        X = np.array(X)
        X_indices = metab.index[rus.sample_indices_]
        OUT_DIR = OUT_DIR + "DENOVO_" + myconfig.model_name + "_" + datestamp + "_undersamping" + "/"
    else:
        labels_dict = y.to_dict()
        labels_dict = {k: int(v) for k, v in labels_dict.items()}
        features_name = metab.columns
        y = np.array(y)
        X = np.array(metab)
        X_indices = metab.index
        OUT_DIR = OUT_DIR + "DENOVO_" + myconfig.model_name + "_" + datestamp + "/"

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    # annotation for metabolite-level data
    annotation_file = "../data/chemical_annotation.tsv"
    annotation_df = pd.read_table(annotation_file)
    annotation_df = annotation_df.set_index("ANALYSIS_ID")
    # cross-validation
    folds=myconfig.n_folds
    fold_v_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [], 'N_epoch': [],
                          'Loss': [], 'N_features': [], 'homophily_index': []}
    fold_test_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [],
                             'N_epoch': []}
    features_track = {'Fold': [], 'N_selected_features': [], 'Selected_Features': [], 'Relevant_Features': []}
    for fold, (train_msk, test_msk, val_msk) in enumerate(zip(*k_fold(X, y, folds))):
        # define data splits
        X_train, X_val, X_test = X[train_msk], X[val_msk], X[test_msk]
        y_train, y_val, y_test = y[train_msk], y[val_msk], y[test_msk]
        scaler = StandardScaler()
        selecter = SelectFromModel(estimator=LogisticRegression(C=myconfig.C_lasso,
                                      penalty="l1",
                                      tol=1e-3,
                                      max_iter=10_000,
                                      solver='liblinear',
                                      random_state=42))
        # for each epoch:
        # fit scaler
        X_train = scaler.fit_transform(X_train)
        # fit feature selection
        selecter.fit(X_train, y_train)
        # feat_names = selecter.get_feature_names_out(features_name)
        feat_mask = selecter.get_support()
        feat_names = features_name[feat_mask].tolist()
        print("number of features", len(feat_names))
        # apply scaler & selecter
        X_processed = selecter.transform(scaler.transform(X))
        X_processed = pd.DataFrame(data=X_processed, columns=feat_names, index= X_indices)
        # build network
        adj = similarity_network(s_threshold=myconfig.S_threshold, X_df=X_processed)
        display_graph(fold, adj, labels_dict, save_fig=True, path=OUT_DIR,
                      name_file=sweep_run.name + "-" + str(fold) + "_network.png",
                      plot_title=sweep_run.name + "- network - fold " + str(fold))
        # create graph data object
        data = create_pyg_data(adj, X_processed, y, train_msk, val_msk, test_msk)
        # Calculate the homophily ratio for the 'label' attribute (edge homophily ratio described in "Beyond Homophily in Graph Neural Networks: Current Limitations
        # and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper
        homophily_index = homophily(data.edge_index, data.y, method='edge')
        # model
        model = generate_model(myconfig.model_name, myconfig, data.num_node_features)
        model = model.to(device)
        # compute class weights for loss function
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(data.y[data.train_mask]),
                                                          y=data.y[data.train_mask].numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=myconfig.lr, weight_decay=myconfig.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=myconfig.lrscheduler_factor,
                                                   threshold=0.0001, patience=15,
                                                   verbose=True)
        n_epochs = myconfig.n_epochs
        if "MLP" in model._get_name():
            # training for non graph methods
            losses, performance, best_epoch, best_loss, best_model = training_mlp(device, model, optimizer, scheduler,
                                                                                  criterion, data, n_epochs, fold)
        else:
            # training for graph methods
            losses, performance, best_epoch, best_loss, best_model = training(device, model, optimizer,
                                                                                          scheduler, criterion, data,
                                                                                          n_epochs, fold) # , embeddings
        # feature importance
        chem_names_list = list(annotation_df.loc[feat_names]["CHEMICAL_NAME"])
        # feature importance
        feature_importance, node_importance = calculate_feature_importance(model, data, names_list=chem_names_list, save_fig=True,
                                                          name_file=f'{sweep_run.name}-{fold}_feature_importance',
                                                          path=OUT_DIR)
        fold_v_performance, fold_test_performance, features_track = update_overall_metrics(fold, best_epoch, homophily_index, feat_names, feature_importance.index.tolist(),
                                                                                           performance, losses, fold_v_performance, fold_test_performance, features_track)
        # log performance and loss in wandb
        eval_info = {f'best_val_loss-{fold}': losses[best_epoch][1],  # val_loss at best epoch
                     f'best_val_Accuracy-{fold}': performance["Accuracy"][best_epoch][1],
                     f'best_val_AUC-{fold}': performance["AUC"][best_epoch][1],
                     f'best_val_Recall-{fold}': performance["Recall"][best_epoch][1],
                     f'best_val_Specificity-{fold}': performance["Specificity"][best_epoch][1],
                     f'best_val_F1-{fold}': performance["F1"][best_epoch][1],
                     f'best_train_AUC-{fold}': performance["AUC"][best_epoch][0],
                     f'best_test_Accuracy-{fold}': performance["Accuracy"][best_epoch][2],
                     f'best_test_AUC-{fold}': performance["AUC"][best_epoch][2],
                     f'best_test_Recall-{fold}': performance["Recall"][best_epoch][2],
                     f'best_test_Specificity-{fold}': performance["Specificity"][best_epoch][2],
                     f'best_test_F1-{fold}': performance["F1"][best_epoch][2],
                     f'features-{fold}': len(feat_names),
                     f'homophily_index-{fold}': homophily_index
                     }
        wandb.log(eval_info)
        # reset parameters
        print('*resetting model parameters*')
        for name, module in model.named_children():
            module.reset_parameters()
    cv_metrics_to_wandb(fold_v_performance, fold_test_performance)
    print("sweep", sweep_run.name, pd.DataFrame.from_dict(fold_v_performance))
    print("sweep", sweep_run.name, pd.DataFrame.from_dict(fold_test_performance))
    print("sweep", sweep_run.name, pd.DataFrame.from_dict(features_track))
    # exports & plots performance & losses
    pd.DataFrame.from_dict(features_track).to_csv(OUT_DIR + sweep_run.name + "_features_track.csv", index=False)
    pd.DataFrame.from_dict(fold_test_performance).to_csv(OUT_DIR + sweep_run.name + "_test_performance.csv", index=False)
    pd.DataFrame.from_dict(fold_v_performance).to_csv(OUT_DIR + sweep_run.name + "_val_performance.csv", index=False)
#wandb.agent(sweep_id, main)
