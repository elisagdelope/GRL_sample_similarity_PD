import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import os
from utils import *
from models import *
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import networkx as nx
from pathlib import Path
from torch_geometric.utils import homophily
import argparse
import shap
from datetime import date
from imblearn.under_sampling import RandomUnderSampler

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = check_cuda()
undersampling = False
myconfig = {'project': 'my-cv-test5',
                'method': 'grid',
                'program': 'cv_test.py',
                'metric': {'name': 'val_loss', 'goal': 'minimize'},
                'parameters': {'C_lasso': {'value': 0.1},
                               'n_folds': {'value': 10},
                               'edge_percent': {'value': 0.4}, # 0.0018, 1
                               'model_name': {'value': "GPST_GINE_lin"},
                               'n_epochs': {'value': 250},
                               'lr': {'value': 0.001},
                               'lrscheduler_factor': {'value': 0.9},
                               'dropout': {'value': 0.3},
                               'weight_decay': {'value': 0.005},
                               'cl1_hidden_units': {'value': 8},
                               'cl2_hidden_units': {'value': 8},
                               'll_out_units': {'value': 2},
                               'K_cheby': {'value':2},
                               'heads': {'value':3}}}


# I/O
OUT_DIR = "../results/"
datestamp = date.today().strftime('%Y%m%d')
OUT_DIR = OUT_DIR + "rnet_" + myconfig['parameters']['model_name']['value'] + "_" + datestamp + "/"
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
folds=myconfig['parameters']['n_folds']['value']
fold_v_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [], 
                        'N_epoch': [], 'Loss': [], 'N_features': [], 'homophily_index': []}
fold_test_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [],
                            'N_epoch': []}
features_track = {'Fold': [], 'N_selected_features': [], 'Selected_Features': [], 'Relevant_Features': []}
for fold, (train_msk, test_msk, val_msk) in enumerate(zip(*k_fold(X, y, folds))):
    # define data splits
    X_train, X_val, X_test = X[train_msk], X[val_msk], X[test_msk]
    y_train, y_val, y_test = y[train_msk], y[val_msk], y[test_msk]
    scaler = StandardScaler()
    selecter = SelectFromModel(estimator=LogisticRegression(C=myconfig['parameters']['C_lasso']['value'],
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
    X_processed = pd.DataFrame(data=X_processed, columns=feat_names, index=X_indices)

    # build network
    #adj = random_network(edge_percentage=myconfig['parameters']['edge_percent']['value'], X_df=X_processed)
    adj = fullyconn_network(X_df=X_processed)
    G = nx.from_pandas_adjacency(adj)
    # plot network
    display_graph(fold, G, pos, labels_dict, save_fig=False, path=OUT_DIR,
                    name_file=str(fold) + "_network.png",
                    plot_title="network - fold " + str(fold), wandb_log=False)
    # create graph data object
    data = create_pyg_data(adj, X_processed, y, train_msk, val_msk, test_msk)
    if "GTC_uw" in myconfig['parameters']['model_name']['value']:
        data.edge_attr = torch.ones((data.edge_index.shape[1], 1), device=data.edge_index.device)
    elif "GTC" in myconfig['parameters']['model_name']['value'] or "GINE" in myconfig['parameters']['model_name']['value']:
        data.edge_attr = data.edge_attr.unsqueeze(-1)
    if "GPST" in myconfig['parameters']['model_name']['value']:
        data.x, feat_names  = pad_features(data.x, myconfig['parameters']['heads']['value'], feat_names)
    # Calculate the homophily ratio for the 'label' attribute (edge homophily ratio described in "Beyond Homophily in Graph Neural Networks: Current Limitations
    # and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper
    homophily_index = homophily(data.edge_index, data.y, method='edge')
    print(f'Homophily index: {homophily_index}')

    # model
    models_dict = {"MLP2": lambda:  MLP2(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GCNN_uw": lambda: GCNN_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GCNN": lambda: GCNN(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "Cheb_GCNN_uw": lambda: Cheb_GCNN_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['K_cheby']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "Cheb_GCNN": lambda: Cheb_GCNN(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['K_cheby']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GAT": lambda: GAT(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['heads']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GAT_uw": lambda: GAT_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['heads']['value'],
                        myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GCNN_10L_uw": lambda: GCNN_10L_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GCNN_10L": lambda: GCNN_10L(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GAT_10L_uw": lambda: GAT_10L_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['heads']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GAT_10L": lambda: GAT_10L(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['heads']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "Cheb_GCNN_10L_uw": lambda: Cheb_GCNN_10L_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['K_cheby']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "Cheb_GCNN_10L": lambda: Cheb_GCNN_10L(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['K_cheby']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GPST_GINE_lin": lambda: GPST_GINE_lin(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'],  myconfig['parameters']['heads']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value'], data.edge_attr.shape[1]),
            "MeanAggMPNN_uw": lambda: MeanAggMPNN_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GTC": lambda: GTC(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['heads']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GTC_uw": lambda: GTC_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['heads']['value'],
                        myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value'])
    }
    model = models_dict[myconfig['parameters']['model_name']['value']]()
    model.apply(init_weights)
    model = model.to(device)
    # compute class weights for loss function
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                        classes=np.unique(data.y[data.train_mask]),
                                                        y=data.y[data.train_mask].numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=myconfig['parameters']['lr']['value'], weight_decay=myconfig['parameters']['weight_decay']['value'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=myconfig['parameters']['lrscheduler_factor']['value'],
                                                threshold=0.0001, patience=15,
                                                verbose=True)
    n_epochs = myconfig['parameters']['n_epochs']['value']
    if "MLP" in model._get_name():
        data.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        # training for non graph methods
        losses, performance, best_epoch, best_loss, best_model = training_mlp_nowandb(device, model, optimizer, scheduler, criterion, data, n_epochs, fold)
        # feature importance
        # feature_importance = feature_importances_shap_values(model, data, X_processed, device, names_list=feat_names, n=20, save_fig=True, name_file=f'{fold}_feature_importance',path=OUT_DIR)
        # feature_importance = list(feature_importance.feature)
    else:
        # training for graph methods
        losses, performance, best_epoch, best_loss, best_model = training_nowandb(device, model, optimizer,scheduler, criterion, data, n_epochs, fold) # , embeddings
    # feature importance
    if "GPST" in myconfig['parameters']['model_name']['value']:
        valid_feat_names = [name for name in feat_names if not name.startswith('pad_feature')] # remove padded features for annotation
        chem_names_list = list(annotation_df.loc[valid_feat_names]["CHEMICAL_NAME"])
        chem_names_list += [name for name in feat_names if name.startswith('pad_feature')] # add padded features
    else: 
        chem_names_list = list(annotation_df.loc[feat_names]["CHEMICAL_NAME"])
    feature_importance = feature_importance_gnnexplainer(model, data, names_list=chem_names_list, save_fig=False, name_file=f'{fold}_feature_importance',path=OUT_DIR)
    feature_importance = feature_importance.index.tolist()
    fold_v_performance, fold_test_performance, features_track = update_overall_metrics(fold, best_epoch, homophily_index, feat_names, feature_importance, performance, losses, fold_v_performance, fold_test_performance, features_track)
    # reset parameters
    print('*resetting model parameters*')
    for name, module in model.named_children():
        module.reset_parameters()
print(pd.DataFrame.from_dict(fold_v_performance))
print(pd.DataFrame.from_dict(fold_test_performance))
print(pd.DataFrame.from_dict(features_track))

# exports & plots performance & losses
pd.DataFrame.from_dict(features_track).to_csv(OUT_DIR + "features_track.csv", index=False)
pd.DataFrame.from_dict(fold_test_performance).to_csv(OUT_DIR + "test_performance.csv", index=False)
pd.DataFrame.from_dict(fold_v_performance).to_csv(OUT_DIR + "val_performance.csv", index=False)



# tests ---------------------
model = GCNN(data.num_node_features, 32, 32, 2, 0.2)
model = GCNN_10L(data.num_node_features, 32, 2, 0.3)
model = MLP2(data.num_node_features, 32, 32, 2, 0.3)
model = GAT_10L(data.num_node_features, 32, 2, 2, 0.3)
model = GUNet_uw(data.num_node_features, 32, 32, 8,8, 2, 2, 0.2)
model = GAT_50L_uw(data.num_node_features, 32, 2, 2, 0.3)
model = Cheb_GCNN_50L(data.num_node_features, 32, 2, 2, 0.3)
model = GAT_50L_uw(data.num_node_features, 32, 2, 2, 0.3)
model = Cheb_GCNN_50L(data.num_node_features, 32, 2, 2, 0.3)

for name, module in model.named_children():
    module.reset_parameters()
model = GPST_GINE_lin(data.num_node_features, 8, 8, 2, 2, 0.2, data.edge_attr.shape[1])
model = GINE(data.num_node_features, 8,8,8,8, 2, 0.2, data.edge_attr.shape[1])
model.apply(init_weights)
model = model.to(device)
# compute class weights for loss function
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(data.y[data.train_mask]),
                                                    y=data.y[data.train_mask].numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9,
                                            threshold=0.0001, patience=15,
                                            verbose=True)
n_epochs = 50

#losses, performance, best_epoch, best_loss, best_model = training_mlp_nowandb(device, model, optimizer,scheduler, criterion, data, n_epochs, fold) 

losses, performance, best_epoch, best_loss, best_model = training_nowandb(device, model, optimizer,scheduler, criterion, data, n_epochs, fold) 

for m in performance.keys():
    print(m, performance[m][best_epoch])
    
feature_importance = feature_importance_gnnexplainer(model, data, names_list=feat_names, save_fig=True, name_file=f'{fold}_feature_importance',path=OUT_DIR)



for fold, (train_msk, test_msk, val_msk) in enumerate(zip(*k_fold(X, y, folds))):
    # define data splits
    X_train, X_val, X_test = X[train_msk], X[val_msk], X[test_msk]
    y_train, y_val, y_test = y[train_msk], y[val_msk], y[test_msk]
    scaler = StandardScaler()
    # fit scaler
    X_train = scaler.fit_transform(X_train)
    # no ft selection here
    feat_names = features_name
    print("number of features", len(feat_names))
    # apply scaler
    X_processed = scaler.transform(X)
    X_processed = pd.DataFrame(data=X_processed, columns=feat_names, index=X_indices)
    # build network
    adj = similarity_network(s_threshold=myconfig['parameters']['S_threshold']['value'], X_df=X_processed)
    G = nx.from_pandas_adjacency(adj)
    # plot network
    display_graph(fold, G, pos, labels_dict, save_fig=False, path=OUT_DIR,
                    name_file=str(fold) + "_network.png",
                    plot_title="network - fold " + str(fold), wandb_log=False)
    # create graph data object
    data = create_pyg_data(adj, X_processed, y, train_msk, val_msk, test_msk)
    # Calculate the homophily ratio for the 'label' attribute (edge homophily ratio described in "Beyond Homophily in Graph Neural Networks: Current Limitations
    # and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper
    homophily_index = homophily(data.edge_index, data.y, method='edge')
    print(f'Homophily index: {homophily_index}')

    # model
    models_dict = {"MLP2": MLP2(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GCNN_uw": GCNN_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GCNN": GCNN(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "Cheb_GCNN_uw": Cheb_GCNN_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['K_cheby']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "Cheb_GCNN": Cheb_GCNN(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['K_cheby']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GAT": GAT(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['heads']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GAT_uw": GAT_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['cl2_hidden_units']['value'], myconfig['parameters']['heads']['value'],
                        myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GCNN_10L_uw": GCNN_10L_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GCNN_10L": GCNN_10L(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GAT_10L_uw": GAT_10L_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['heads']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "GAT_10L": GAT_10L(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['heads']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "Cheb_GCNN_10L_uw": Cheb_GCNN_10L_uw(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['K_cheby']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value']),
            "Cheb_GCNN_10L": Cheb_GCNN_10L(data.num_node_features, myconfig['parameters']['cl1_hidden_units']['value'], myconfig['parameters']['K_cheby']['value'], myconfig['parameters']['ll_out_units']['value'], myconfig['parameters']['dropout']['value'])

    }
    model = models_dict[myconfig['parameters']['model_name']['value']]
    model.apply(init_weights)
    model = model.to(device)
    # compute class weights for loss function
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                        classes=np.unique(data.y[data.train_mask]),
                                                        y=data.y[data.train_mask].numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=myconfig['parameters']['lr']['value'], weight_decay=myconfig['parameters']['weight_decay']['value'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=myconfig['parameters']['lrscheduler_factor']['value'],
                                                threshold=0.0001, patience=15,
                                                verbose=True)
    n_epochs = myconfig['parameters']['n_epochs']['value']
    if "MLP" in model._get_name():
        data.edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        # training for non graph methods
        losses, performance, best_epoch, best_loss, best_model = training_mlp_nowandb(device, model, optimizer, scheduler, criterion, data, n_epochs, fold)
        # feature importance
        # feature_importance = feature_importances_shap_values(model, data, X_processed, device, names_list=feat_names, n=20, save_fig=True, name_file=f'{fold}_feature_importance',path=OUT_DIR)
        # feature_importance = list(feature_importance.feature)
    else:
        # training for graph methods
        losses, performance, best_epoch, best_loss, best_model = training_nowandb(device, model, optimizer,scheduler, criterion, data, n_epochs, fold) # , embeddings
        # feature importance
    feature_importance = feature_importance_gnnexplainer(model, data, names_list=feat_names, save_fig=True, name_file=f'{fold}_feature_importance',path=OUT_DIR)
    feature_importance = feature_importance.index.tolist()
    fold_v_performance, fold_test_performance, features_track = update_overall_metrics(fold, best_epoch, homophily_index, feat_names, feature_importance, performance, losses, fold_v_performance, fold_test_performance, features_track)
    # reset parameters
    print('*resetting model parameters*')
    for name, module in model.named_children():
        module.reset_parameters()
