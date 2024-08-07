import seaborn as sns
from matplotlib import pyplot as plt
import shap
import textwrap
import numpy as np

def training_plots (Nepochs, values_train, values_val, var_name, save_fig=False, name_file=None, path=None):
    array_epochs = range(1, Nepochs+1)

    # Plot and label the training and validation loss values
    plt.plot(array_epochs, values_train, label='Training ' + var_name)
    plt.plot(array_epochs, values_val, label='Validation ' + var_name)

    # Add in a title and axes labels
    plt.title('Training and Validation ' + var_name)
    plt.xlabel('Epochs')
    plt.ylabel(var_name)

    # Set the tick locations
    plt.xticks(range(0, Nepochs, (1 if Nepochs<40 else int(Nepochs/20))), fontsize=8)

    # Display the plot
    plt.legend(loc='best')
    plt.show()
    if save_fig:
        plt.savefig(path + name_file)
    plt.close()


def pca_plot2d(v1, v2, color_labels, save_fig=False, name_file=None, path=None):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=v1, y=v2,
        hue=color_labels,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.8
    )
    if save_fig:
        plt.savefig(path + name_file)
    plt.close()

def pca_plot3d(v1, v2, v3, color_labels, save_fig=False, name_file=None, path=None):
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=v1,
        ys=v2,
        zs=v3,
        c=color_labels,
        cmap='viridis'
    )
    ax.set_xlabel('pca-v1')
    ax.set_ylabel('pca-v2')
    ax.set_zlabel('pca-v3')
    plt.show()
    if save_fig:
        plt.savefig(path + name_file)
    plt.close()


def plot_from_shap_values(shap_values, X, save_fig=False, name_file=None, path=None, names_list=None, plot_title=None):
    """Plot shap values for a given scikit-learn model.

    Parameters
    ----------
    shap_values : array, shape (n_samples, n_features)
                  Shap values matrix
    X : array, shape (n_samples, n_features).
        Training data.
    save_fig : bool
               Whether or not to save the figures.
    name_file : str
                Name of the file if saved.
    path : str
           Path for the saved figure.
    names_list : list
                 Names of the features (length # features) to appear in the plot.
    plot_title : str
                 Title of the plot.
    """

    if not names_list:
        names_list = list(X.columns)

    shap_fig = shap.summary_plot(shap_values=shap_values, features=X, feature_names=names_list, show=False)
    #    shap.plots.beeswarm(shap_values, show=False)
    fig, ax = plt.gcf(), plt.gca()
    labels = ax.get_yticklabels()
    fig.axes[-1].set_aspect(120)  # set the color bar
    fig.axes[-1].set_box_aspect(120)  # set the color bar
    fig.axes[-1].set_ylabel('Feature value', fontsize=13, labelpad=-25)
    fig.set_size_inches(11, 6.5)  # set figure size
    wrapped_labels = [textwrap.fill(str(label.get_text()), 50) for label in labels]
    ax.set_yticklabels(wrapped_labels, fontsize=13)
    ax.set_xticklabels(np.round(ax.get_xticks(), 2), rotation=15, fontsize=11)
    plt.title(plot_title)
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    # image = Image.frombytes('RGB', fig.canvas.get_width_height(),
    #                         fig.canvas.tostring_rgb())
    # wandb.log({f'shap-{fold}': wandb.Image(image), "caption": "Shap values analysis"})
    if save_fig:
        fig.savefig(path + name_file)
        plt.close(fig)
    else:
       fig.show()
