import os
import pandas as pd

# I/O
results_dir = '../results/wandb'
output_dir = '../results'

# Initialize lists to store the average and std dataframes
df_averages = []
df_stds = []

# Walk through the results directory
for experiment_name in os.listdir(results_dir):
    if experiment_name != 'mytests':
        experiment_path = os.path.join(results_dir, experiment_name)
        cv_file_path = os.path.join(experiment_path, 'cv-results', 'cv_test_results.csv')
        
        # Check if the path is a directory and the results file exists
        if os.path.isdir(experiment_path) and os.path.isfile(cv_file_path):
            # Read the CSV file
            df = pd.read_csv(cv_file_path)
            
            # Extract the average and std rows and make a copy to avoid SettingWithCopyWarning
            avg_row = df[df['Fold'] == 'average'].copy()
            std_row = df[df['Fold'] == 'std'].copy()

            # Replace the 'Fold' column with the experiment name
            avg_row['Fold'] = experiment_name
            std_row['Fold'] = experiment_name

            # Rename the 'Fold' column to 'Experiment'
            avg_row.rename(columns={'Fold': 'Experiment'}, inplace=True)
            std_row.rename(columns={'Fold': 'Experiment'}, inplace=True)

            # Append the rows to the lists
            df_averages.append(avg_row)
            df_stds.append(std_row)

# Concatenate the lists into dataframes
df_averages = pd.concat(df_averages)
df_stds = pd.concat(df_stds)

# Sort by 'Experiment' alphabetically
df_averages = df_averages.sort_values(by='Experiment')
df_stds = df_stds.sort_values(by='Experiment')

# Save the dataframes to new CSV files
df_averages.to_csv(os.path.join(output_dir, 'psn_cv_test_avg.csv'),index=False)
df_stds.to_csv(os.path.join(output_dir, 'psn_cv_test_std.csv'), index=False)

print(f"Patient similarity network AVG and STD dataframes created and saved in {output_dir} directory.")