import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xml.etree.ElementTree as ET
import os

def parse_gxl(file_path):
    """ Parse a GXL file and return a NetworkX graph. """
    graph = nx.Graph()
    tree = ET.parse(file_path)
    root = tree.getroot()
    for node in root.findall('.//node'):
        node_id = node.get('id')
        symbol = node.find('attr').find('string').text
        graph.add_node(node_id, symbol=symbol)
    for edge in root.findall('.//edge'):
        node1 = edge.get('from')
        node2 = edge.get('to')
        graph.add_edge(node1, node2)
    return graph

def compute_cost_matrix(graph1, graph2):
    """ Compute the cost matrix for bipartite graph matching. """
    nodes1 = list(graph1.nodes(data=True))
    nodes2 = list(graph2.nodes(data=True))
    
    # Initialize cost matrix
    cost_matrix = np.zeros((len(nodes1), len(nodes2)))
    
    # Fill cost matrix
    for i, (n1, d1) in enumerate(nodes1):
        for j, (n2, d2) in enumerate(nodes2):
            if d1['symbol'] != d2['symbol']:
                cost_matrix[i, j] = 2  # Node substitution cost
            else:
                cost_matrix[i, j] = 0
    
    return cost_matrix

def compute_ged(graph1, graph2):
    """ Compute GED using bipartite graph matching. """
    cost_matrix = compute_cost_matrix(graph1, graph2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind].sum()
    return cost

def load_data(gxl_folder, train_file, validation_file, test_file):
    """ Load data from files. """
    train_graphs = []
    train_labels = []
    validation_graphs = []
    validation_labels = []
    test_graphs = []
    test_ids = []
    
    with open(train_file, 'r') as f:
        for line in f:
            graph_id, label = line.strip().split('\t')
            train_graphs.append(parse_gxl(os.path.join(gxl_folder, f'{graph_id}.gxl')))
            train_labels.append(label)
    
    with open(validation_file, 'r') as f:
        for line in f:
            graph_id, label = line.strip().split('\t')
            validation_graphs.append(parse_gxl(os.path.join(gxl_folder, f'{graph_id}.gxl')))
            validation_labels.append(label)
    
    with open(test_file, 'r') as f:
        for line in f:
            graph_id = line.strip()
            test_graphs.append(parse_gxl(os.path.join(gxl_folder, f'{graph_id}.gxl')))
            test_ids.append(graph_id)
    
    return train_graphs, train_labels, validation_graphs, validation_labels, test_graphs, test_ids

def extract_features(train_graphs, test_graph):
    """ Extract features for a test graph against all training graphs. """
    features = []
    for train_graph in train_graphs:
        ged = compute_ged(train_graph, test_graph)
        features.append(ged)
    return features

def main():
    gxl_folder = 'gxl'
    train_file = 'train.tsv'
    validation_file = 'validation.tsv'
    test_file = 'test.tsv'
    
    train_graphs, train_labels, validation_graphs, validation_labels, test_graphs, test_ids = load_data(gxl_folder, train_file, validation_file, test_file)
    
    # Extract features for training and validation sets
    train_features = np.array([extract_features(train_graphs, train_graph) for train_graph in train_graphs])
    validation_features = np.array([extract_features(train_graphs, val_graph) for val_graph in validation_graphs])
    
    # Convert labels to binary
    train_labels = np.where(np.array(train_labels) == 'active', 1, 0)
    validation_labels = np.where(np.array(validation_labels) == 'active', 1, 0)
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)
    
    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = knn.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    
    # Predict test set
    test_features = np.array([extract_features(train_graphs, test_graph) for test_graph in test_graphs])
    y_test_pred = knn.predict(test_features)
    
    # Output predictions to test.tsv
    with open('test.tsv', 'w') as f:
        for test_id, label in zip(test_ids, y_test_pred):
            f.write(f"{test_id} {'active' if label == 1 else 'inactive'}\n")

if __name__ == "__main__":
    main()