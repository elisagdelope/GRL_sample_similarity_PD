import numpy as np
import pandas as pd
import random
import os
from matplotlib import pyplot as plt
from utils import *
from models import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
import networkx as nx
import wandb
import yaml
from pathlib import Path
from torch_geometric.utils import homophily
import argparse
import shap
from datetime import date
from imblearn.under_sampling import RandomUnderSampler
import matplotlib 

# Set matplotlib to 'agg' 
if matplotlib.get_backend() != 'agg':
    print(f"Switching Matplotlib backend from '{matplotlib.get_backend()}' to 'agg'")
    matplotlib.use('agg')

#----------------  Main function -----------------#
if __name__ == '__main__':
    # set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='sweep_config', type=str)  # , default='configs/default.yaml'
    args, unknown = parser.parse_known_args()
    # set up wandb
    sweep_run = wandb.init(config=args.sweep_config)
    myconfig = wandb.config
    print('Config file from wandb:', myconfig)
    # set other params
    device = check_cuda()
    undersampling = False
    # I/O
    OUT_DIR = "../results/wandb/"
    datestamp = date.today().strftime('%Y%m%d')
    OUT_DIR = OUT_DIR + "fcnet_" + myconfig.model_name + "_" + datestamp + "/" # change vs cv_wandb.py
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    metab_file = "../data/data_metab_4ML_DIAGNOSIS.tsv"
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
        pos = get_pos_similarity(X)
        y = np.array(y)
        X = np.array(X)
        X_indices = metab.index[rus.sample_indices_]
    else:
        labels_dict = y.to_dict()
        labels_dict = {k: int(v) for k, v in labels_dict.items()}
        features_name = metab.columns
        pos = get_pos_similarity(metab)
        y = np.array(y)
        X = np.array(metab)
        X_indices = metab.index
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
        X_processed = pd.DataFrame(data=X_processed, columns=feat_names, index=metab.index) 
        # build network
        adj = fullyconn_network(X_df=X_processed)  # change vs cv_wandb.py
        G = nx.from_pandas_adjacency(adj)
        # plot network
        display_graph(fold, G, pos, labels_dict, save_fig=True, path=OUT_DIR,
                      name_file=sweep_run.name + "-" + str(fold) + "_network.png",
                      plot_title=sweep_run.name + "- network - fold " + str(fold), wandb_log=True)
        # create graph data object
        data = create_pyg_data(adj, X_processed, y, train_msk, val_msk, test_msk)
        if "GTC_uw" in myconfig.model_name:
            data.edge_attr = torch.ones((data.edge_index.shape[1], 1), device=data.edge_index.device)
        elif "GTC" in myconfig.model_name:
            data.edge_attr = data.edge_attr.unsqueeze(-1)
        if "GPST" in myconfig.model_name:
            data.x, feat_names  = pad_features(data.x, myconfig.heads, feat_names)
        # Calculate the homophily ratio for the 'label' attribute (edge homophily ratio described in "Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper
        homophily_index = homophily(data.edge_index, data.y, method='edge')
        print(f'Homophily index: {homophily_index}')
        # model
        print(myconfig.model_name)
        model = generate_model(myconfig.model_name, myconfig, data)
        model.apply(init_weights)
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
            data.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            # training for non graph methods
            losses, performance, best_epoch, best_loss, best_model = training_mlp(device, model, optimizer, scheduler, criterion, data, n_epochs, fold, wandb)
            # feature importance
            # chem_names_list = list(annotation_df.loc[feat_names]["CHEMICAL_NAME"])
            # feature_importance = feature_importances_shap_values(model, data, X_processed, device, names_list=chem_names_list, n=20, save_fig=True, name_file=f'{sweep_run.name}-{fold}_feature_importance', path=OUT_DIR)
            # feature_importance = list(feature_importance.feature)
        else:
            # training for graph methods 
            losses, performance, best_epoch, best_loss, best_model = training(device, model, optimizer,
                                                                            scheduler, criterion, data,
                                                                            n_epochs, fold, wandb) # , embeddings
        # feature importance
        if "GPST" in myconfig.model_name:
            valid_feat_names = [name for name in feat_names if not name.startswith('pad_feature')] # remove padded features for annotation
            chem_names_list = list(annotation_df.loc[valid_feat_names]["CHEMICAL_NAME"])
            chem_names_list += [name for name in feat_names if name.startswith('pad_feature')] # add padded features
        else: 
            chem_names_list = list(annotation_df.loc[feat_names]["CHEMICAL_NAME"])
        # feature importance   
        feature_importance = feature_importance_gnnexplainer(model, data, names_list=chem_names_list, save_fig=True, name_file=f'{sweep_run.name}-{fold}_feature_importance',path=OUT_DIR)
        feature_importance = feature_importance.index.tolist()
        fold_v_performance, fold_test_performance, features_track = update_overall_metrics(fold, best_epoch, homophily_index, feat_names, feature_importance, performance, losses, fold_v_performance, fold_test_performance, features_track)
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
            if isinstance(module, torch.nn.ModuleList):
                for sub_module in module:
                    if hasattr(sub_module, 'reset_parameters'):
                        sub_module.reset_parameters()
            else:
                if hasattr(module, 'reset_parameters'):
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


