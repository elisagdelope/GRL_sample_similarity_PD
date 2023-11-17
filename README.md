Scripts to perform graph representation learning modelling on sample-similarity networks of transcriptomics and metabolomics data from the PPMI and the luxPARK cohort, respectively. 

### Data pre-processing, prior to ML modelling

#### ppmi_analyses 

##### ppmi_data4ML_class.R
This script performs unsupervised filters to generate data for ML modelling of snapshot data (T0) from RNAseq data.

* ppmi_data4ML_class.R employs as input transcriptomics and phenotypical data resulting from previous pre-processing scripts described in repository *statistical_analyses_cross_long_PD* for **parsing data** and **Baseline (T0) PD/HC** (ppmi_filter_gene_expression.R, ppmi_norm_gene_expression.R, ppmi_generate_pathway_level.R). 

#### luxpark_analyses 

lx_data4ML_class.R, lx_data4ML_class_denovo.R
This script performs unsupervised filters to generate data for ML modelling of snapshot data (T0) from metabolomics data (all PD/HC and *de novo* PD/HC).


### Modelling

##### cv_wandb_4ML.py
Main script to perform the training, hyperparameter tunning and cross-validation of GCN, ChebyNet and GAT models using molecular interaction networks to classify the omics profiles as signals on a graph (i.e., graph classification). The script requires a config file for the hyperparameter search (.yaml). Weights & biases are used to monitor the training.

##### cv_results.py
Extracts the cross-validated results by looking at the minimum validation loss and generates figures and results tables based on node and edge importance.

##### features_plot.py
Generates barplot with most relevant nodes and their functional annotation.
