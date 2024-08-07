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
from plot_utils import plot_from_shap_values
from models import *
import shap
from torch.autograd import Variable
from torch_geometric.explain import Explainer, GNNExplainer
import torch.nn.init as init
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import warnings
import os
import tarfile
import random 

# --------------------------- set up------------------------------
def check_cuda():
    if torch.cuda.is_available():
        print('cuda available')
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        device = torch.device('cuda')
        torch.cuda.manual_seed(42)
    else:
        print('cuda not available')
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        device = torch.device('cpu')
    return device

# ------------------------ graph creation ------------------------
class MyPSN(InMemoryDataset):
    """
    MyPSN dataset class for processing and loading patient network data.
    Args:
        root (string): Root directory where the dataset should be saved.
        X_file (string): File path to the gene expression data (embeddings).
        graph_file (string): File path to the adjacency matrix.
        labels_cv_file (string): File path to the cross-validation labels.
        labels_test_file (string): File path to the test labels.
        transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in an
            `torch_geometric.data.Data` object and returns a transformed version.
            The data object will be transformed before being saved to disk.
        pre_filter (callable, optional): A function that takes in an
            `torch_geometric.data.Data` object and returns a boolean value,
            indicating whether the data object should be included in the dataset.
    Attributes:
        X_file (string): File path to the gene expression data (embeddings).
        graph_file (string): File path to the adjacency matrix.
        labels_cv_file (string): File path to the cross-validation labels.
        labels_test_file (string): File path to the test labels.
        data (torch_geometric.data.Data): The processed data object.
        slices (dict): A dictionary holding the assignment of the dataset to
            the different splits (train, test, etc.).
    """
    def __init__(self, root, X_file, graph_file, labels_cv_file, labels_test_file, transform=None, pre_transform=None, pre_filter=None):
        """
        Initializes a new instance of the MyPSN dataset.
        Args:
            root (string): Root directory where the dataset should be saved.
            X_file (string): File path to the gene expression data (embeddings).
            graph_file (string): File path to the adjacency matrix.
            labels_cv_file (string): File path to the cross-validation labels.
            labels_test_file (string): File path to the test labels.
            transform (callable, optional): A function/transform that takes in an
                `torch_geometric.data.Data` object and returns a transformed version.
                The data object will be transformed before every access.
            pre_transform (callable, optional): A function/transform that takes in an
                `torch_geometric.data.Data` object and returns a transformed version.
                The data object will be transformed before being saved to disk.
            pre_filter (callable, optional): A function that takes in an
                `torch_geometric.data.Data` object and returns a boolean value,
                indicating whether the data object should be included in the dataset.
        """
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


def is_symmetric(matrix: np.ndarray) -> bool:
    return np.array_equal(matrix, matrix.T)


def find_unconnected(adj_matrix):
    """
    Finds the nodes in an adjacency matrix that have no connections.
    Parameters:
    adj_matrix (list of lists): The adjacency matrix representing the connections between nodes.
    Returns:
    list: A list of nodes that have no connections.
    """
    # Create a list of all the nodes
    nodes = list(range(len(adj_matrix)))
    # Use a lambda function to check if a row in the adjacency matrix has any non-zero entries
    has_connections = lambda row: any(x != 0 for x in row)

    # Use the filter function to find the nodes with no connections
    unconnected_nodes = list(filter(lambda i: not has_connections(adj_matrix[i]), nodes))
    return unconnected_nodes


def add_edge_to_unconnected_nodes(sim_matrix, adj_matrix):
    """
    Adds an edge to each unconnected node in the adjacency matrix based on the strongest connection from the similarity matrix.
    Args:
        sim_matrix (pd.DataFrame): The similarity matrix.
        adj_matrix (pd.DataFrame): The adjacency matrix.
    Returns:
        pd.DataFrame: The updated adjacency matrix with added edges.
    """
    unconnected = find_unconnected(np.array(adj_matrix))
    print("There are %d unconnected nodes for which to add edges" % len(unconnected))

    # Remove self-loops from similarity matrix if any
    sim_matrix = sim_matrix.mask(np.eye(sim_matrix.shape[0], dtype=bool))

    # Find the node with the strongest edge for unconnected nodes
    strongest_edges = sim_matrix.iloc[unconnected, :].max(axis=1)
    idx_strongest = sim_matrix.iloc[unconnected, :].idxmax(axis=1)

    # Update the adjacency matrix with the strongest edges for unconnected nodes
    for node, edge_strength, strongest_node in zip(unconnected, strongest_edges, idx_strongest):
        node_idx = adj_matrix.index[node]
        adj_matrix.loc[node_idx, strongest_node] = edge_strength
        adj_matrix.loc[strongest_node, node_idx] = edge_strength

    # Print the minimum similarity of the strongest edge for unconnected nodes
    print("The weakest edge that was added for unconnected nodes has a similarity of %f" % strongest_edges.min())

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
        print(f'The weakest edge that was added for unconnected components (subgraphs) has a similarity of {min(strength_track)}.')
        # explore the new graph
        describe_graph(adj_df)
    else:
        print("The graph is connected; no need to add edges")
    return adj_df


def describe_graph(adj_df):
    """
    Analyzes a graph represented by an adjacency matrix.
    Parameters:
    adj_df (pandas.DataFrame): The adjacency matrix as a pandas DataFrame.
    Returns:
    None
    Prints the following information about the graph:
    - Number of nodes in the graph
    - Number of edges in the graph
    - Number of non-zero elements in the adjacency matrix
    - Whether the graph is directed or not
    - Whether the graph is connected or not
    - Whether the graph is weighted or not
    - Number of connected components in the graph
    - Size of the largest connected component
    - Degree distribution of the graph
    """
    G = nx.from_pandas_adjacency(adj_df)
    print("N nodes in G:", len(G.nodes))
    print("N edges in G:", len(G.edges))
    print("N non-zero elements in adjacency matrix:", np.count_nonzero(np.array(adj_df)), "out of", np.square(len(G.nodes)), "entries")  # make sure it should be 2* number of edges or [2*number_edges - number_nodes] if self-loops exist
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
        adj_df = add_edge_to_unconnected_nodes(sim, adj_df)
        print("avg adj", np.mean(np.array(adj_df)))
        if not is_symmetric(np.array(adj_df)):
            raise ValueError('Adjacency matrix is not symmetric')
    # explore the graph
    G = nx.from_pandas_adjacency(adj_df)
    describe_graph(adj_df)
    # check for unconnected components and connect them with the minimum number of the strongest links
    adj_df = connect_components_with_strongest_links(adj_df, sim)
    return adj_df


def random_network(edge_percentage, X_df, max_iter_er=1000, tolerance=0.01):
    """
    Create a random, connected Erdős–Rényi network with a specified percentage of edges.
    Parameters:
        - edge_percentage (float): Percentage of possible edges to include in the network (between 0 and 1).
        - X_df (pandas DataFrame): DataFrame containing data points as rows and features as columns.he network.
        - max_iter_er (int): Maximum number of iterations to attempt generating a connected ER graph.
        - tolerance (float): Allowed tolerance for the number of edges as a percentage of target edges in ER graph.
    Returns:
        - adj_df (pandas DataFrame): Adjacency matrix representing the random network.
    """
    num_nodes = X_df.shape[0]
    max_edges = num_nodes * (num_nodes - 1) // 2  # Max number of edges in an undirected graph without self-loops
    target_edges = int(edge_percentage * max_edges)
    fallback = False
    if target_edges <= (num_nodes - 1):
        print("Edge percentage is too low to create a connected graph. Fallback to a random chain.")
        fallback = True
    else: 
        # Create a random, connected Erdős–Rényi graph
        p = edge_percentage  # ER probability for edge creation
        for _ in range(max_iter_er):
            G = nx.erdos_renyi_graph(num_nodes, p)
            if nx.is_connected(G): 
                if abs(G.number_of_edges() - target_edges) <= int(tolerance * target_edges): # If the number of edges ~= target, convert G to adj and return adj matrix
                    adj_matrix = nx.to_numpy_array(G) 
                    adj_df = pd.DataFrame(adj_matrix, index=X_df.index, columns=X_df.index) 
                    print("A random, connected Erdős–Rényi was created.")
                    describe_graph(adj_df)
                    return adj_df
            p = random.uniform(p, p*(1 + tolerance))  # Randomly sample a new p value within the tolerance range
        fallback = True

    if fallback:
        # fallback to a random connected graph (random chain + random edges) if ER graph is not connected after max_iter_er
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        # random chain
        nodes = list(range(num_nodes))
        random.shuffle(nodes)
        for i in range(1, num_nodes):
            G.add_edge(nodes[i - 1], nodes[i])
        # Add random edges until the target number of edges is reached
        while G.number_of_edges() < target_edges:
            u, v = random.sample(range(num_nodes), 2)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)

    adj_matrix = nx.to_numpy_array(G)
    adj_df = pd.DataFrame(adj_matrix, index=X_df.index, columns=X_df.index)
    print("A random, connected graph was created (not ER)")
    describe_graph(adj_df)
    return adj_df


def fullyconn_network(X_df):
    """
    Create a fully connected network.
    Parameters:
    X_df (pandas DataFrame): DataFrame containing data points as rows and features as columns.
    Returns:
    adj_df (pandas DataFrame): Adjacency matrix representing the fully connected network.
    """
    num_nodes = X_df.shape[0]
    adj_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    adj_df = pd.DataFrame(adj_matrix, index=X_df.index, columns=X_df.index)
    print("A fully connected graph without self-loops was created.")
    describe_graph(adj_df)
    return adj_df


def calculate_homophily_ratio(G, attr_name):
    """
    Calculate the homophily ratio for a given attribute in a graph.
    The homophily ratio is the ratio of connected pairs of nodes with the same labels with respect to connected pairs of nodes with different labels
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


def get_pos_similarity(X_df):
    """
    Calculate the position of nodes in a similarity graph based on cosine distance.
    Parameters:
    X_df (pandas.DataFrame): The input data frame containing the data points.
    Returns:
    dict: A dictionary representing the positions of the data points in a graph.
    Raises:
    ValueError: If the adjacency matrix is not symmetric.
    """
    # Calculate pairwise cosine distance between data points
    dist = pd.DataFrame(
        squareform(pdist(X_df, metric='cosine')),
        columns=X_df.index,
        index=X_df.index
    )
    # Calculate similarity from distance
    sim = 1 - dist
    sim = sim.fillna(0)
    sim_np = np.array(sim)
    sim_np[np.diag_indices_from(sim_np)] = 0.  # remove self-loops
    sim = pd.DataFrame(sim_np, index=sim.index, columns=sim.columns)
    if not is_symmetric(np.array(sim)):
        raise ValueError('Adjacency matrix is not symmetric')
    G = nx.from_pandas_adjacency(sim)
#   pos = nx.spring_layout(G, seed=42)
    pos = nx.spring_layout(G, seed=42)
    return pos


def display_graph(fold, G, pos, labels_dict=None, save_fig=False, path="./", name_file="graph.png", plot_title=None, wandb_log=True):
    """
    Draw the graph given a networkx graph G and a set of positions.
    Parameters:
    - fold (int): The fold number.
    - G (networkx.Graph): The graph object.
    - pos (dict): A dictionary mapping node IDs to positions.
    - labels_dict (dict, optional): A dictionary mapping node IDs to labels. Defaults to None.
    - save_fig (bool, optional): Whether to save the graph as an image. Defaults to False.
    - path (str, optional): The path to save the image. Defaults to "./".
    - name_file (str, optional): The name of the saved image file. Defaults to "graph.png".
    - plot_title (str, optional): The title of the graph plot. Defaults to None.
    - wandb_log (bool, optional): Whether to log the image to wandb. Defaults to True.
    """

    fig = plt.figure(figsize=(12, 12))
    weights = nx.get_edge_attributes(G, 'weight').values()
    if min(weights) == max(weights):
        # If all weights are the same, set them to a constant value (1.5)
        weights = [1.5] * len(weights) 
    else:
        # Normalize weights to the range [0.5, 5]
        weights = [(weight - min(weights)) * (5 - 0.5) / (max(weights) - min(weights)) + 0.5 for weight in weights]
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
        # Get the colors of edges:
        edge_colors = []
        for u, v in G.edges():
            u_label = G.nodes[u].get("label")
            v_label = G.nodes[v].get("label")
            if u_label == v_label:
                edge_color = "gray"  # Same class edges
            else:
                edge_color = "darkorange"    # Different class edges
            edge_colors.append(edge_color)
        # Draw the graph
        nx.draw(G, pos=pos, with_labels=False,
                cmap=plt.get_cmap("viridis"), node_color=colors, node_size=80, edge_color=edge_colors,
                width=list(weights), ax=fig.add_subplot(111))
    plt.title(plot_title, fontsize=24)
    plt.tight_layout()
    if save_fig:
        fig.savefig(path + name_file)
    # Log the image to wandb: Convert the graph image to a PIL Image
    if wandb_log:
        fig.canvas.draw()  # Force the rendering of the figure
        image = Image.frombytes('RGB', fig.canvas.get_width_height(),
                            fig.canvas.tostring_rgb())
        wandb.log({f'graph-{fold}': wandb.Image(image), "caption": "Graph Visualization"})
    plt.close(fig)


def create_pyg_data(adj_df, X_df, y, train_msk, val_msk, test_msk):
    """
    Create a PyTorch Geometric Data object from the given inputs.
    Args:
        adj_df (pandas.DataFrame): The adjacency matrix as a DataFrame.
        X_df (pandas.DataFrame): The feature matrix as a DataFrame.
        y (numpy.ndarray): The target labels as a numpy array.
        train_msk (torch.Tensor): The training mask as a boolean tensor.
        val_msk (torch.Tensor): The validation mask as a boolean tensor.
        test_msk (torch.Tensor): The test mask as a boolean tensor.
    Returns:
        torch_geometric.data.Data: The PyTorch Geometric Data object.
    """
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


def pad_features(x, num_heads, feat_names):
    """
    Pad the input feature matrix `x` to make its dimension divisible by `num_heads`.

    Args:
        x (torch.Tensor): The input feature matrix of shape (n, d), where n is the number of nodes and d is the number of features.
        num_heads (int): The number of attention heads in the Multi-Head Attention mechanism.
        feat_names (list of str): The list of feature names corresponding to the columns of `x`.

    Returns:
        torch.Tensor: The padded feature matrix of shape (n, d'), where d' is the padded dimension of the features.
        list of str: The padded list of feature names.
    """
    embed_dim = x.size(1)  # Get the current dimension of the input features
    remainder = embed_dim % num_heads  # Calculate the remainder when embed_dim is divided by num_heads

    if remainder != 0:
        # If the remainder is not zero, we need to pad the features
        pad_dim = num_heads - remainder  # Calculate the required padding size
        pad_tensor = torch.zeros((x.size(0), pad_dim), device=x.device)  # Create a new tensor filled with zeros for padding
        x = torch.cat([x, pad_tensor], dim=1)  # Concatenate the original features with the padding tensor along the feature dimension

        # Pad the feature names list with placeholder names for the new padded features
        padded_feat_names = feat_names + [f'pad_feature_{i}' for i in range(pad_dim)]
    else:
        # If no padding is needed, return the original feature names
        padded_feat_names = feat_names

    return x, padded_feat_names



# --------- Cross-validation & metrics ----------

def k_fold(x, y, folds):
    """
    Splits the data into k folds for cross-validation.
    Args:
        x (torch.Tensor): The input data.
        y (numpy.ndarray): The target labels.
        folds (int): The number of folds for cross-validation.
    Returns:
        tuple: A tuple containing the train mask, test mask, and validation mask for each fold.
    """
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
    return train_mask, test_mask, val_mask


def init_weights(m):
    """
    Initializes the weights of a linear layer using Xavier uniform initialization.
    Args:
        m (torch.nn.Linear): The linear layer to initialize.
    """
    if isinstance(m, torch.nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)


def generate_model(model_name, config, data):
    """
    Generate a model based on the given model name.
    Args:
        model_name (str): The name of the model to generate.
        config (object): The configuration object containing the model parameters.
        n_features (int): The number of input features.
    Returns:
        model: The generated model.
    Raises:
        KeyError: If the given model name is not found in the models_dict.
    """
    if hasattr(data, 'edge_attr') and data.edge_attr is not None and len(data.edge_attr.shape) > 1:
        edge_dim = data.edge_attr.shape[1]
    else:
        edge_dim = 0
    n_features = data.num_node_features
    models_dict = { # Dictionary of model names and their corresponding lambda functions to instantiate the models only when they are actually needed.  This approach avoids initializing all models upfront.
        "MLP2": lambda: MLP2(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout),
        "GCNN_uw": lambda: GCNN_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout),
        "GCNN": lambda: GCNN(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout),
        "Cheb_GCNN_uw": lambda: Cheb_GCNN_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.K_cheby, config.ll_out_units, config.dropout),
        "Cheb_GCNN": lambda: Cheb_GCNN(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.K_cheby, config.ll_out_units, config.dropout),
        "GAT": lambda: GAT(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout),
        "GAT_uw": lambda: GAT_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout),
        "GCNN_10L_uw": lambda: GCNN_10L_uw(n_features, config.cl1_hidden_units, config.ll_out_units, config.dropout),
        "GCNN_10L": lambda: GCNN_10L(n_features, config.cl1_hidden_units, config.ll_out_units, config.dropout),
        "Cheb_GCNN_10L_uw": lambda: Cheb_GCNN_10L_uw(n_features, config.cl1_hidden_units, config.K_cheby, config.ll_out_units, config.dropout),
        "Cheb_GCNN_10L": lambda: Cheb_GCNN_10L(n_features, config.cl1_hidden_units, config.K_cheby, config.ll_out_units, config.dropout),
        "Cheb_GCNN_50L": lambda: Cheb_GCNN_50L(n_features, config.cl1_hidden_units, config.K_cheby, config.ll_out_units, config.dropout),
        "GAT_10L_uw": lambda: GAT_10L_uw(n_features, config.cl1_hidden_units, config.heads, config.ll_out_units, config.dropout),
        "GAT_50L_uw": lambda: GAT_50L_uw(n_features, config.cl1_hidden_units, config.heads, config.ll_out_units, config.dropout),
        "GAT_10L": lambda: GAT_10L(n_features, config.cl1_hidden_units, config.heads, config.ll_out_units, config.dropout),
        "MeanAggMPNN_uw": lambda: MeanAggMPNN_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout),
        "GTC": lambda: GTC(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout),
        "GTC_uw": lambda: GTC_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout),
        "GPST_lin": lambda: GPST_lin(n_features, config.cl1_hidden_units, config.heads, config.ll_out_units, config.dropout, config.K_cheby),
        "GPST": lambda: GPST(n_features, config.heads, config.ll_out_units, config.dropout, config.K_cheby),
        "GPST_GINE": lambda: GPST_GINE(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout, edge_dim),
        "GPST_GINE_lin": lambda: GPST_GINE_lin(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout, edge_dim),
        "GINE": lambda: GINE(n_features, config.h1_hidden_units, config.cl1_hidden_units, config.h2_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout, edge_dim),
        "GUNet_uw": lambda: GUNet_uw(n_features, config.h1_hidden_units, config.cl1_hidden_units, config.h2_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.depth, config.dropout),
        "GUNet": lambda: GUNet(n_features, config.h1_hidden_units, config.cl1_hidden_units, config.h2_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.depth, config.dropout)
    }
    if model_name not in models_dict:
        raise KeyError(f"Model name '{model_name}' is not found in the models_dict.")
    model = models_dict[model_name]()  # Call the lambda function to instantiate the model
    print(model)
    return model

def update_overall_metrics(fold, fold_best_epoch, homophily_index, feat_names, relevant_features, fold_performance, fold_losses, dict_val_metrics, dict_test_metrics, features_track):
    """
    Update the overall metrics with the results from a fold.
    Args:
        fold (int): The fold number.
        fold_best_epoch (int): The best epoch for the fold.
        homophily_index (float): The homophily index for the fold.
        feat_names (list): The list of feature names.
        relevant_features (list): The list of relevant features.
        fold_performance (dict): The performance metrics for each epoch in the fold.
        fold_losses (dict): The loss values for each epoch in the fold.
        dict_val_metrics (dict): The dictionary to store validation metrics.
        dict_test_metrics (dict): The dictionary to store test metrics.
        features_track (dict): The dictionary to track selected features.
    Returns:
        tuple: A tuple containing the updated dictionaries dict_val_metrics, dict_test_metrics, and features_track.
    """
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
    """
    Logs cross-validation metrics to Weights & Biases (wandb) for visualization and analysis.
    Args:
        dict_val_metrics (dict): A dictionary containing validation metrics.
        dict_test_metrics (dict): A dictionary containing test metrics.
    Returns:
        None
    """
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


def get_results(results_dict):
    """
    Calculate summary statistics from a dictionary of results.
    Parameters:
    results_dict (dict): A dictionary containing the results data.
    Returns:
    pandas.DataFrame: A DataFrame containing the summary statistics.
    """
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



# ------------ training & evaluation ------------
def train_epoch(device, model, optimizer, criterion, data, metric):
    """Train step of model on training dataset for one epoch.
    Args:
        device (torch.device): The device to perform the training on.
        model (torch.nn.Module): The model to train.
        data (torch_geometric.data.Data): The training data.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
        metric (torchmetrics.Metric): The metric for evaluating the model performance.
    Returns:
        tuple: A tuple containing the training loss and the training accuracy.
    """
    model.to(device)
    model.train()
    data.to(device)
    criterion.to(device)
    optimizer.zero_grad()  # Clear gradients
    model_name = str(model.__class__.__name__)
    # Perform a single forward pass
    if ("GAT" in model_name or "GTC" in model_name or "GINE" in model_name) and not model_name == "GAT_uw": # GAT, GTC, GTC_uw, GINE, GPST_GINE_lin, GPST_GINE
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    elif "_uw" in model_name or "GPST" in model_name:  # for unweighted models exceot GTC_uw
        y_hat = model(x=data.x, edge_index=data.edge_index)
    else:
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32)) # GCN, CHEBYNET, GUNet    
    # # Perform a single forward pass
    # if "_uw" in str(model.__class__.__name__) and not str(model.__class__.__name__) == "GTC_uw":  # for unweighted models
    #     y_hat = model(x=data.x, edge_index=data.edge_index)
    # elif "GAT" in str(model.__class__.__name__) or "GTC" in str(model.__class__.__name__):
    #     y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    # else:
    #     y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32))
    loss = criterion(y_hat[data.train_mask], data.y[data.train_mask])  # Compute the loss
    loss.backward()  # Derive gradients
    optimizer.step()  # Update parameters based on gradients
    # track loss & embeddings
    tloss = loss.detach().cpu().numpy().item()
    # track performance
    y_hat = y_hat[:,1]  # get label
    batch_acc = metric(y_hat[data.train_mask].cpu(), data.y[data.train_mask].cpu())
    train_acc = metric.compute()
    return tloss, train_acc


def evaluate_epoch(device, model, criterion, data, metric):
    """Evaluate the model on validation data for a single epoch.
    Args:
        device (torch.device): The device to perform the evaluation on.
        model (torch.nn.Module): The model to evaluate.
        data (torch_geometric.data.Data): The validation data.
        criterion (torch.nn.Module): The loss criterion.
        metric (torchmetrics.Metric): The evaluation metric.
    Returns:
        tuple: A tuple containing the validation loss and the validation accuracy.
    """
    model.eval()
    model.to(device)
    data.to(device)
    criterion.to(device)
    model_name = str(model.__class__.__name__)
    # Perform a single forward pass
    if ("GAT" in model_name or "GTC" in model_name or "GINE" in model_name) and not model_name == "GAT_uw": # GAT, GTC, GTC_uw, GINE, GPST_GINE_lin, GPST_GINE
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    elif "_uw" in model_name or "GPST" in model_name:  # for unweighted models exceot GTC_uw
        y_hat = model(x=data.x, edge_index=data.edge_index)
    else:
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32)) # GCN, CHEBYNET, GUNet  
    if "val_mask" in data.items()._keys():
        vloss = criterion(y_hat[data.val_mask], data.y[data.val_mask])  # Compute the loss
        vloss = vloss.detach().cpu().numpy().item()
        y_hat = y_hat[:, 1]
        batch_vacc = metric(y_hat[data.val_mask].cpu(), data.y[data.val_mask].cpu())
    else:
        vloss = criterion(y_hat[data.test_mask], data.y[data.test_mask])  # Compute the loss
        vloss = vloss.detach().cpu().numpy().item()
        y_hat = y_hat[:, 1]
        batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())
    val_acc = metric.compute()
    return vloss, val_acc


def test_epoch(device, model, data, metric):
    """Evaluate the model on test data for a single epoch.
    Args:
        device (torch.device): The device to perform the evaluation on.
        model (torch.nn.Module): The model to evaluate.
        data (torch_geometric.data.Data): The test data.
        metric (torchmetrics.Metric): The metric to compute the performance.
    Returns:
        float: The test accuracy.
    """
    model.eval()
    data.to(device)
    model_name = str(model.__class__.__name__)
    # Perform a single forward pass
    if ("GAT" in model_name or "GTC" in model_name or "GINE" in model_name) and not model_name == "GAT_uw": # GAT, GTC, GTC_uw, GINE, GPST_GINE_lin, GPST_GINE
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    elif "_uw" in model_name or "GPST" in model_name:  # for unweighted models exceot GTC_uw
        y_hat = model(x=data.x, edge_index=data.edge_index)
    else:
        y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32)) # GCN, CHEBYNET, GUNet  
    y_hat = y_hat[:, 1]  # get label
    batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())
    test_acc = metric.compute()
    return test_acc


def training(device, model, optimizer, scheduler, criterion, data, n_epochs, fold, wandb):
    """ Full training process, logs in wandb.
    Args:
        device (torch.device): The device to run the training on.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        criterion (torch.nn.Module): The loss function.
        data (DataLoader): The data used for training.
        n_epochs (int): The number of training epochs.
        fold (int): The fold number.
    Returns:
        tuple: A tuple containing the following elements:
            - losses (list): A list of training and validation losses for each epoch.
            - perf_metrics (dict): A dictionary containing performance metrics for each epoch.
            - best_epoch (int): The epoch number with the best validation loss.
            - best_loss (float): The best validation loss.
            - best_model (torch.nn.Module): The model with the best validation loss.
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
    test_metrics = MetricCollection({
                'Accuracy': Accuracy(task="binary"),
                'AUC': AUROC(task="binary", num_classes=2),
                'Recall': Recall(task="binary", num_classes=2),
                'Specificity': Specificity(task="binary", num_classes=2),
                'F1': F1Score(task="binary", num_classes=2),
    })
    # # Define the custom x axis metric
    wandb.define_metric(f'epoch_fold-{fold}')
    # Define which metrics to plot against that x-axis
    wandb.define_metric(f'val/loss-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/loss-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/Accuracy-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/AUC-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/Recall-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/Specificity-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/F1-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/Accuracy-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/AUC-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/Recall-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/Specificity-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/F1-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/AUC-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/Accuracy-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/Recall-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/Specificity-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/F1-{fold}', step_metric=f'epoch_fold-{fold}')
    step_fold=0
    for epoch in range(n_epochs):
        step_fold +=0
        # train
        train_loss, train_perf = train_epoch(device, model, optimizer, criterion, data, train_metrics) #, epoch_embeddings
        # validation
        val_loss, val_perf = evaluate_epoch(device, model, criterion, data, val_metrics)
        # scheduler step
        scheduler.step(val_loss)
        # track losses & embeddings
        losses.append([train_loss, val_loss])
        #embeddings.append(epoch_embeddings)
        test_perf = test_epoch(device, model, data, test_metrics)
        for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(), test_perf[m].detach().numpy().item()])
        # log performance and loss in wandb
        wandb.log({f'epoch_fold-{fold}': epoch,
                    f'val/loss-{fold}': val_loss,
                   f'train/loss-{fold}': train_loss,
                   f'val/Accuracy-{fold}': val_perf["Accuracy"].detach().numpy().item(),
                   f'val/AUC-{fold}': val_perf["AUC"].detach().numpy().item(),
                   f'val/Recall-{fold}': val_perf["Recall"].detach().numpy().item(),
                   f'val/Specificity-{fold}': val_perf["Specificity"].detach().numpy().item(),
                   f'val/F1-{fold}': val_perf["F1"].detach().numpy().item(),
                   f'train/Accuracy-{fold}': train_perf["Accuracy"].detach().numpy().item(),
                   f'train/AUC-{fold}': train_perf["AUC"].detach().numpy().item(),
                   f'train/Recall-{fold}': train_perf["Recall"].detach().numpy().item(),
                   f'train/Specificity-{fold}': train_perf["Specificity"].detach().numpy().item(),
                   f'train/F1-{fold}': train_perf["F1"].detach().numpy().item(),
                   f'test/AUC-{fold}': test_perf["AUC"].detach().numpy().item(),
                   f'test/Accuracy-{fold}': test_perf["AUC"].detach().numpy().item(),
                   f'test/Recall-{fold}': test_perf["Recall"].detach().numpy().item(),
                   f'test/Specificity-{fold}': test_perf["Specificity"].detach().numpy().item(),
                   f'test/F1-{fold}': test_perf["F1"].detach().numpy().item()
                   }) #, step=epoch)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}",
                  f"Loss train {train_loss}",
                  f"Loss validation {val_loss}",
                  f"Acc train {train_perf}",
                  f"Acc validation {val_perf};")
        train_metrics.reset()
        val_metrics.reset()
        test_metrics.reset()

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
        torch.cuda.empty_cache()
    return losses, perf_metrics, best_epoch, best_loss, best_model #, embeddings


def training_nowandb(device, model, optimizer, scheduler, criterion, data, n_epochs, fold):
    """Performs the full training process without logging in wandb.
    Args:
        device (torch.device): The device to be used for training.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        criterion (torch.nn.Module): The loss function used for training.
        data (torch.utils.data.Dataset): The dataset used for training.
        n_epochs (int): The number of training epochs.
        fold (int): The fold number.
    Returns:
        tuple: A tuple containing the following elements:
            - losses (list): A list of training and validation losses for each epoch.
            - perf_metrics (dict): A dictionary containing performance metrics (Accuracy, AUC, Recall, Specificity, F1) for each epoch.
            - best_epoch (int): The epoch number with the best validation loss.
            - best_loss (float): The best validation loss.
            - best_model (torch.nn.Module): The model with the best validation loss.
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
    test_metrics = MetricCollection({
                'Accuracy': Accuracy(task="binary"),
                'AUC': AUROC(task="binary", num_classes=2),
                'Recall': Recall(task="binary", num_classes=2),
                'Specificity': Specificity(task="binary", num_classes=2),
                'F1': F1Score(task="binary", num_classes=2),
    })
    for epoch in range(n_epochs):
        # train
        train_loss, train_perf = train_epoch(device, model, optimizer, criterion, data, train_metrics) #, epoch_embeddings
        # validation
        val_loss, val_perf = evaluate_epoch(device, model, criterion, data, val_metrics)
        # scheduler step
        scheduler.step(val_loss)
        # track losses & embeddings
        losses.append([train_loss, val_loss])
        #embeddings.append(epoch_embeddings)
        test_perf = test_epoch(device, model, data, test_metrics)
        for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(), test_perf[m].detach().numpy().item()])
        if epoch % 5 == 0:
            print(f"Epoch {epoch}",
                  f"Loss train {train_loss}",
                  f"Loss validation {val_loss}",
                  f"Acc train {train_perf}",
                  f"Acc validation {val_perf};")
        train_metrics.reset()
        val_metrics.reset()
        test_metrics.reset()

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
    """
    Trains a multi-layer perceptron (MLP) model.
    Args:
        device (torch.device): The device to perform the training on.
        model (torch.nn.Module): The MLP model to train.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss criterion used for training.
        data (torch_geometric.data.Data): The input data for training.
        metric (torchmetrics.Metric): The metric used for evaluating the model performance.
    Returns:
        tuple: A tuple containing the epoch loss (float) and the training performance (float).
    """
    model.train()
    data.to(device)
    optimizer.zero_grad()  # Clear gradients.
    y_hat = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(y_hat[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    # track loss & embeddings
    epoch_loss = loss.detach().cpu().numpy().item()
    y_hat = y_hat[:,1]
    batch_perf = metric(y_hat[data.train_mask].cpu(), data.y[data.train_mask].cpu())
    train_perf = metric.compute()
    return epoch_loss, train_perf


def evaluate_mlp(device, model, criterion, data, metric):
    """
    Evaluates the performance of a multi-layer perceptron (MLP) model on the given data.
    Args:
        device (torch.device): The device to perform the evaluation on.
        model (torch.nn.Module): The MLP model to evaluate.
        criterion: The loss criterion used for evaluation.
        data: The data to evaluate the model on.
        metric: The performance metric used for evaluation.
    Returns:
        tuple: A tuple containing the validation loss and the validation performance.
    """
    model.eval()
    data.to(device)
    y_hat = model(data.x, data.edge_index) #_, y_hat
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
    """
    Evaluate the performance of a multi-layer perceptron (MLP) model on test data.
    Args:
        device (torch.device): The device to run the model on.
        model (torch.nn.Module): The MLP model to evaluate.
        data (torch_geometric.data.Data): The input data for the model.
        metric (torch.nn.Module): The metric to compute the performance.
    Returns:
        float: The performance of the model on the test data.
    """
    model.eval()
    data.to(device)
    y_hat = model(data.x, data.edge_index) # _, y_hat
    y_hat = y_hat[:,1]
    batch_perf = metric(y_hat[data.test_mask].cpu(), data.y[data.test_mask].cpu())
    test_perf = metric.compute()
    return test_perf


def training_mlp(device, model, optimizer, scheduler, criterion, data, n_epochs, fold, wandb):
    """
    Trains a multi-layer perceptron (MLP) model using the specified device, optimizer, scheduler, criterion, data, and number of epochs. Logs in wandb.
    Args:
        device (torch.device): The device to be used for training (e.g., 'cuda' for GPU or 'cpu' for CPU).
        model (torch.nn.Module): The MLP model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler used for adjusting the learning rate during training.
        criterion (torch.nn.Module): The loss function used for computing the training loss.
        data (torch.utils.data.Dataset): The dataset used for training, validation, and testing.
        n_epochs (int): The number of training epochs.
        fold (int): The fold number for tracking the performance and loss.
    Returns:
        tuple: A tuple containing the following elements:
            - losses (list): A list of training and validation losses for each epoch.
            - perf_metrics (dict): A dictionary containing performance metrics (e.g., accuracy, AUC, recall, specificity, F1) for each epoch.
            - best_epoch (int): The epoch number with the lowest validation loss.
            - best_loss (float): The lowest validation loss achieved during training.
            - best_model (torch.nn.Module): The model with the lowest validation loss.

    """
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
    test_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    # Define the custom x axis metric
    wandb.define_metric(f'epoch_fold-{fold}')
    # Define which metrics to plot against that x-axis
    wandb.define_metric(f'val/loss-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/loss-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/Accuracy-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/AUC-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/Recall-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/Specificity-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/F1-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/Accuracy-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/AUC-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/Recall-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/Specificity-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/F1-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/AUC-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/Accuracy-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/Recall-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/Specificity-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/F1-{fold}', step_metric=f'epoch_fold-{fold}')
    for epoch in range(n_epochs):
        # train
        train_loss, train_perf = train_mlp(device, model, optimizer, criterion, data, train_metrics)
        # validation
        val_loss, val_perf = evaluate_mlp(device, model, criterion, data, val_metrics)
        # scheduler step
        scheduler.step(val_loss)
        # track losses & embeddings
        losses.append([train_loss, val_loss])
        #embeddings.append(epoch_embeddings)
        test_perf = test_mlp(device, model, data, test_metrics)
        for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(), test_perf[m].detach().numpy().item()])
        # log performance and loss in wandb
        wandb.log({f'epoch_fold-{fold}': epoch,
                    f'val/loss-{fold}': val_loss,
                   f'train/loss-{fold}': train_loss,
                   f'val/Accuracy-{fold}': val_perf["Accuracy"].detach().numpy().item(),
                   f'val/AUC-{fold}': val_perf["AUC"].detach().numpy().item(),
                   f'val/Recall-{fold}': val_perf["Recall"].detach().numpy().item(),
                   f'val/Specificity-{fold}': val_perf["Specificity"].detach().numpy().item(),
                   f'val/F1-{fold}': val_perf["F1"].detach().numpy().item(),
                   f'train/Accuracy-{fold}': train_perf["Accuracy"].detach().numpy().item(),
                   f'train/AUC-{fold}': train_perf["AUC"].detach().numpy().item(),
                   f'train/Recall-{fold}': train_perf["Recall"].detach().numpy().item(),
                   f'train/Specificity-{fold}': train_perf["Specificity"].detach().numpy().item(),
                   f'train/F1-{fold}': train_perf["F1"].detach().numpy().item(),
                   f'test/AUC-{fold}': test_perf["AUC"].detach().numpy().item(),
                   f'test/Accuracy-{fold}': test_perf["AUC"].detach().numpy().item(),
                   f'test/Recall-{fold}': test_perf["Recall"].detach().numpy().item(),
                   f'test/Specificity-{fold}': test_perf["Specificity"].detach().numpy().item(),
                   f'test/F1-{fold}': test_perf["F1"].detach().numpy().item()
                   }) #, step=epoch)
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


def training_mlp_nowandb(device, model, optimizer, scheduler, criterion, data, n_epochs, fold):
    """
    Trains a multi-layer perceptron (MLP) model without logging in wandb.
    Args:
        device (torch.device): The device to run the training on.
        model (torch.nn.Module): The MLP model to train.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        criterion (torch.nn.Module): The loss function.
        data (torch.utils.data.DataLoader): The data loader for training, validation, and testing data.
        n_epochs (int): The number of training epochs.
        fold (int): The fold number.
    Returns:
        tuple: A tuple containing the following elements:
            - losses (list): A list of training and validation losses for each epoch.
            - perf_metrics (dict): A dictionary of performance metrics (accuracy, AUC, recall, specificity, F1) for each epoch.
            - best_epoch (int): The epoch number with the best validation loss.
            - best_loss (float): The best validation loss.
            - best_model (torch.nn.Module): The best model with the lowest validation loss.
    """
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
    test_metrics = MetricCollection({
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
        #embeddings.append(epoch_embeddings)
        test_perf = test_mlp(device, model, data, test_metrics)
        for m in perf_metrics.keys():
                perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(), test_perf[m].detach().numpy().item()])
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



# ------------ feature importance ------------


def get_feature_importance(explanation, feat_labels=None, top_k=None):
    """
    Calculates feature importance scores from the explanation object.

    Args:
        explanation: The explanation object containing feature importance information.
        feat_labels (List[str], optional): List of feature names. (default: None)
        top_k (int, optional): Number of top features to return. (default: None, returns all)

    Returns:
        pandas.DataFrame: A DataFrame containing two columns:
            - 'Feature Label': Top feature labels (as a column), set as the index.
            - 'Importance_score': Corresponding importance scores.
    """

    node_mask = explanation.get('node_mask')
    if node_mask is None:
        raise ValueError(f"The attribute 'node_mask' is not available "
                          f"in '{explanation.__class__.__name__}' "
                          f"(got {explanation.available_explanations})")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f"Cannot compute feature importance for "
                          f"object-level 'node_mask' "
                          f"(got shape {node_mask.size()})")

    if feat_labels is None:
        feat_labels = range(node_mask.size(1))

    score = node_mask.sum(dim=0).cpu().numpy()

    if top_k is not None:
        top_features_labels = feat_labels[:top_k]
        importance_scores = score[:top_k]
    else:
        top_features_labels = feat_labels
        importance_scores = score

    df_top_features = pd.DataFrame({
        'Importance_score': importance_scores
    }, index= top_features_labels)
    return df_top_features


def feature_importance_gnnexplainer(model, data, names_list=None, save_fig=False, name_file='feature_importance', path=None, n=20):
    """
    Calculate the feature importance using the GNN-Explainer model.
    Args:
        model (torch.nn.Module): The GNN model.
        data (torch_geometric.data.Data): The input data.
        names_list (list, optional): List of feature names. Defaults to None.
        save_fig (bool, optional): Whether to save the feature importance plot and subgraph visualization plot. Defaults to False.
        name_file (str, optional): The name of the saved files. Defaults to 'feature_importance'.
        path (str, optional): The path to save the files. Defaults to None.
        n (int, optional): The number of top features to visualize. Defaults to 20.

    Returns:
        tuple: A tuple containing the feature importance plot and the subgraph visualization plot.
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
    model_name = str(model.__class__.__name__)
    if "MLP" in model_name:  # for unweighted models
        try:
            explanation = explainer(x=data.x, edge_index=data.edge_index)
        except Exception as e:
            print(f"Catched exception: {e}")
            explanation = explainer(x=data.x, edge_index=data.edge_index)
    elif ("GAT" in model_name or "GTC" in model_name or "GINE" in model_name) and not model_name == "GAT_uw": # GAT, GTC, GTC_uw, GINE, GPST_GINE_lin, GPST_GINE
        explanation = explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr.to(torch.float32))
    elif "_uw" in model_name or "GPST" in model_name:  # for unweighted models exceot GTC_uw
        explanation = explainer(x=data.x, edge_index=data.edge_index) 
    else:
        explanation = explainer(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr.to(torch.float32)) # GCN, CHEBYNET, GUNet       
    print(f'Generated explanations in {explanation.available_explanations}')
    if save_fig:
        if path is None:
            path = os.getcwd() + "/"
        feat_importance = explanation.visualize_feature_importance(str(path) + name_file + ".png",
                                                                   top_k=n, feat_labels=names_list)
        print(f"Feature importance plot has been saved to '{path}'")
        feat_importance = get_feature_importance(explanation, names_list, top_k=n)
        #node_importance = explanation.visualize_graph(path + name_file + "_subgraph.pdf")
        #print(f"Subgraph visualization plot has been saved to '{path}'")
    else:
        feat_importance = explanation.visualize_feature_importance(path=None,
                                                                   top_k=n, feat_labels=names_list)
        feat_importance = get_feature_importance(explanation, names_list, top_k=n)
        #node_importance = explanation.visualize_graph(path=None)
    return feat_importance #, node_importance




def feature_importances_shap_values(model, data, X, device, names_list=None, n=20, save_fig=False, name_file='feature_importance', path=None):
    """
    Extracts the top n relevant features based on SHAP values in an ordered way
    Parameters:
        model (torch.nn.Module): The trained PyTorch model.
        data (torch.Tensor): The input data for the model.
        X (pandas.DataFrame): The feature matrix.
        names_list (list, optional): The list of feature names. If not provided, the column names of X will be used.
        n (int, optional): The number of top features to extract. Default is 20.
    Returns:
        pandas.DataFrame: A DataFrame containing the top n features and their corresponding SHAP values.
    """
    # generate shap values

    # Define function to wrap model to transform data to tensor
    f = lambda x: model(Variable(torch.from_numpy(x).to(device)), data.edge_index).detach().cpu().numpy()

    #explainer = shap.KernelExplainer(f, data.x.cpu().detach().numpy())
    #shap_values = explainer.shap_values(data.x.cpu().detach().numpy()) # takes a long time
    explainer = shap.KernelExplainer(f, shap.sample(data.x.cpu().detach().numpy(), 10))
    warnings.filterwarnings('ignore', 'The default of \'normalize\' will be set to False in version 1.2 and deprecated in version 1.4.*')
    shap_values = explainer.shap_values(data.x.cpu().detach().numpy())
    shap_values = shap_values[1]  # for binary classification
    # convert shap values to a pandas DataFrame
    if not names_list:
        names_list = list(X.columns)
    shap_df = pd.DataFrame(shap_values, columns=names_list)
    vals = np.abs(shap_df).mean(0)
    shap_importance = pd.DataFrame(list(zip(names_list, vals)),
                                   columns=['feature', 'shap_value'])
    shap_importance.sort_values(by=['shap_value'],
                                ascending=False,
                                inplace=True)
    shap_importance = shap_importance.iloc[0:n, ]


    plot_from_shap_values(shap_values, X, save_fig=save_fig, name_file=name_file, path=path, names_list=names_list, plot_title=None)

    return shap_importance


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


class GraphMasker:
    """
    A class for masking graph data.
    Args:
        data (torch_geometric.data.Data): The input graph data.
    Attributes:
        data (torch_geometric.data.Data): The input graph data.
        x (torch.Tensor): The node feature matrix.
        edge_index (torch.Tensor): The edge index matrix.
        shape (torch.Size): The shape of the node feature matrix.
    """
    def __init__(self, data):
        self.data = data
        self.x = data.x
        self.edge_index = data.edge_index
        self.shape = data.x.shape
    def __call__(self, mask):
        """
        Apply the given mask to the node feature matrix.
        Args:
            mask (torch.Tensor): The mask to be applied.
        Returns:
            torch.Tensor: The masked node feature matrix.
        """
        self.x = torch.where(mask[:, None].bool(), torch.zeros_like(self.x), self.x)
        self.data.x = self.x
        self.data.edge_index = self.edge_index
        return self.data.x  # Return the x attribute as a Tensor



def build_sparse_mask(dense_mask, k):
    """
    Builds a sparse mask from a dense mask by adding k connections between each pair of nodes.
    Args:
        dense_mask (numpy.ndarray): The dense mask representing the connections between nodes.
        k (int): The number of connections to add between each pair of nodes.
    Returns:
        tuple: A tuple containing the sparse mask, the number of input features, and the number of output features.
    """
    sparse_mask = torch.tensor(np.array(dense_mask)).nonzero()
    in_features = dense_mask.shape[0]
    out_features = dense_mask.shape[1] * k
    k_connections = torch.tensor([], dtype=int)
    for i in range(1, k):
        temp = copy.deepcopy(dense_mask)
        temp[:, 1] = temp[:, 1] + (dense_mask.shape[1] * i)
        k_connections = torch.cat((k_connections, temp), dim=0)
    sparse_mask = torch.cat((sparse_mask, k_connections), dim=0)
    return sparse_mask, in_features, out_features

def compress_dir(dir):
    dir = os.path.normpath(dir)
    comp_filename = os.path.basename(dir) + "-files.tar.gz"
    comp_file_path = os.path.join(dir, comp_filename)
    with tarfile.open(comp_file_path, 'w:gz') as tar: # Create a tar.gz file for compression
        for entry in os.scandir(dir):
            if entry.is_file(): # Add only files to the tar archive
                tar.add(entry.path, arcname=entry.name)
    return comp_filename

def rm_files(dir):
    for entry in os.scandir(dir):
        if entry.is_file() and not entry.name.endswith('.tar.gz'): # Remove all files except the compressed tar.gz file
            os.remove(entry.path)
