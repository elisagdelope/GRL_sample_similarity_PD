import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  
import pandas as pd
import numpy as np
import argparse
import re

# I/O
parser = argparse.ArgumentParser()
parser.add_argument('file_path', help='Path to cv_overlappingfeature.csv results file')
args = parser.parse_args()
FILE = args.file_path #"../results/wandb/DENOVO_Cheb_GCNN_20230915_undersamping/cv-results/cv_overlappingfeatures.csv"
#FILE = "../results/wandb/DENOVO_Cheb_GCNN_20230915_undersamping/cv-results/cv_overlappingfeatures.csv"
PLOTS_DIR = "../results/plots/"
df = pd.read_csv(FILE, delimiter=',')
model_dir = re.search(r'wandb/(.*?)/cv-results', FILE)
model_dir = model_dir.group(1)

# remove unknown metabolites
df = df[df['SUB_PATHWAY'].notnull()]

# count threshold
x = 3
df = df[df['Count'] >= x]

# Sort the DataFrame by 'Count'
df = df.sort_values(by='Count', ascending=False)

# Generate color codes for each unique sub_pathway
unique_sub_pathways = df['SUB_PATHWAY'].unique()
color_map = plt.get_cmap('tab20')
color_codes = color_map(np.linspace(0, 1, len(unique_sub_pathways)))
color_dict = dict(zip(unique_sub_pathways, color_codes))

# Add line breaks to the long labels
labels = [
    '\n'.join(label.split(' ', 2)[:2]) if len(label) > 20 else label
    for label in df['Feature']
]
# Generate the horizontal bar plot with correct color coding
fig, ax = plt.subplots(figsize=(10, 6))
bars = plt.barh(
    labels,  # Use the modified labels
    df['Count'],
    color=[color_dict[pathway] for pathway in df['SUB_PATHWAY']]
)

# Set plot aesthetics
plt.xlabel("Number of times among top-20 most relevant features (out of 10)")
plt.ylabel("Features")
plt.title("Metabolites by relevant features frequency in 10-fold CV")

# Create custom legend
legend_labels = [Line2D([0], [0], color=color_dict[pathway], lw=4, label=pathway) for pathway in unique_sub_pathways]
legend = plt.legend(handles=legend_labels, title='Corresponding pathways', loc='lower right',  framealpha=0.0,  markerfirst=False)
legend._legend_box.align = "right"
legend.get_title().set_fontweight('bold')

plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig(PLOTS_DIR + model_dir + "_" + 'frequent_features_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(PLOTS_DIR + model_dir + "_" + 'frequent_features_plot.pdf', bbox_inches='tight')
plt.show()



