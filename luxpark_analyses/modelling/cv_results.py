import os
import pandas as pd
import shutil
from collections import Counter
import argparse
import ast
import re
from utils import compress_dir, rm_files
# I/O
parser = argparse.ArgumentParser()
parser.add_argument('dir_path', help='Directory with wandb sweeps results')
args = parser.parse_args()
DIR = args.dir_path #"../results/wandb/Cheb_GCNN_20230428"

# Define the filename pattern to match
val_pattern = "_val_performance.csv"
# Get a list of all files in the folder that match the pattern
val_files = [f for f in os.listdir(DIR) if f.endswith(val_pattern)]
# Initialize an empty dictionary to store the lowest loss values
lowest_losses = {}

# select first 130 sweeps
val_files = sorted(val_files, key=lambda x: int(re.search(r'\d+', x).group() if re.search(r'\d+', x) else float('inf')))[:130]

# Iterate over the files
for file in val_files:
    # Read the file into a pandas DataFrame
    df = pd.read_csv(os.path.join(DIR, file))
    # Loop over each row in the DataFrame
    for index, row in df.iterrows():
        # Get the current row's loss value
        loss = row["Loss"]
        # Check if we already have a lowest loss value for this row
        if index not in lowest_losses or loss < lowest_losses[index]["Loss"]:
            # If not, update the lowest_losses dictionary with the new lowest loss value and the file name
            lowest_losses[index] = {"Loss": loss, "Sweep": file.replace(val_pattern, "")}
            print(index, loss, file)

for index, result in lowest_losses.items():
    print(f"For fold {index}, the sweep with the lowest loss value is {result['Sweep']}")

columns = ['Fold', 'AUC', 'Accuracy', 'Recall', 'Specificity', 'F1', 'N_epoch']
cvperformance_df = pd.DataFrame(columns=columns)
features_df = pd.DataFrame(columns=['Fold', 'Relevant_Features'])
DIR_RESULTS = DIR + "/cv-results"
# Loop over each winner file in the lowest_losses dictionary
# retrieve test performance, relevant features, and graph images
for index, min_loss in lowest_losses.items():
    # test performance
    test_file = [f for f in os.listdir(DIR) if (f.startswith(min_loss["Sweep"]) and f.endswith("test_performance.csv"))][0]
    test_df = pd.read_csv(os.path.join(DIR, test_file))
    cvperformance_df = cvperformance_df.append(test_df.loc[index,columns], ignore_index=True)
    # relevant features
    features_file = [f for f in os.listdir(DIR) if (f.startswith(min_loss["Sweep"]) and f.endswith("features_track.csv"))][0]
    features = pd.read_csv(os.path.join(DIR, features_file))
    features_df = features_df.append(features.loc[index,['Fold', 'Relevant_Features']], ignore_index=True)
    # graph images
    if not os.path.exists(DIR_RESULTS):
        os.makedirs(DIR_RESULTS)
    img_files =[f for f in os.listdir(DIR) if
     (f.startswith(min_loss["Sweep"]) and (f.endswith("-" + str(index) + "_" + "network.png") or f.endswith("-" + str(index) + "_" + "feature_importance.png")))]
    for file in img_files:
        source_path = os.path.join(DIR, file)
        destination_path = os.path.join(DIR_RESULTS, str(index) + "-" + file)
        shutil.copyfile(source_path, destination_path)

mean_row = cvperformance_df.mean()
std_row = cvperformance_df.std()
# append the mean and std rows to the DataFrame
cvperformance_df = cvperformance_df.append(mean_row.rename('average'))
cvperformance_df = cvperformance_df.append(std_row.rename('std'))
cvperformance_df[['Fold', 'N_epoch']] = cvperformance_df[['Fold', 'N_epoch']].astype(int)
cvperformance_df.loc[['average', 'std'],['Fold', 'N_epoch']] = "NA"
cvperformance_df['Fold'] = cvperformance_df.index

# features
feature_sets = [set(features) for features in features_df['Relevant_Features'].apply(ast.literal_eval)]
overlapping_ft = set.intersection(*feature_sets)
all_ft = [feature for features in features_df['Relevant_Features'].apply(ast.literal_eval) for feature in features]
# Count the occurrences of each feature
ft_counts = Counter(all_ft)
#repeated_features = [feature for feature, count in feature_counts.items() if count > 1]
ft_dict = {'Feature': list(ft_counts.keys()), 'Count': list(ft_counts.values())}
ft_df = pd.DataFrame(ft_dict)
ft_df.sort_values(by='Count', ascending=False, inplace=True)

# annotatec
annotation_file = "../data/chemical_annotation.tsv"
chem_annotation = pd.read_csv(annotation_file, sep='\t')
ft_df = pd.merge(ft_df, chem_annotation[['SUPER_PATHWAY','SUB_PATHWAY', "CHEMICAL_NAME"]], left_on='Feature', right_on="CHEMICAL_NAME")
ft_df = ft_df.drop('CHEMICAL_NAME', axis=1)

# export results
cvperformance_df.to_csv(DIR_RESULTS + "/cv_test_results.csv", index=False)
features_df.to_csv(DIR_RESULTS + "/cv_relevantfeatures.csv", index=False)
ft_df.to_csv(DIR_RESULTS + "/cv_overlappingfeatures.csv", index=False)

# compress DIR files
compress_dir(DIR)
rm_files(DIR)




