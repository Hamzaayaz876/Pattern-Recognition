import os
import pandas as pd
from grakel import GraphKernel
from grakel import Graph
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from lxml import etree


def read_gxl_to_grakel(filepath):
    tree = etree.parse(filepath)
    root = tree.getroot()
    graph_elem = root.find("graph")

    nodes = {}
    edges = []

    # Read nodes and labels
    for node in graph_elem.findall("node"):
        node_id = node.get("id")
        label = node.find("attr").find("string").text.strip()
        nodes[node_id] = label

    # Read edges
    for edge in graph_elem.findall("edge"):
        source = edge.get("from")
        target = edge.get("to")
        edges.append((source, target))

    # Map node ids to indices for GraKeL
    id_to_idx = {nid: idx for idx, nid in enumerate(nodes)}
    edge_list = [(id_to_idx[u], id_to_idx[v]) for u, v in edges]
    label_dict = {id_to_idx[k]: v for k, v in nodes.items()}

    return Graph(edge_list, node_labels=label_dict)


def load_graphs(gxl_dir, ids):
    graphs = []
    for mol_id in tqdm(ids, desc="Loading graphs"):
        path = os.path.join(gxl_dir, f"{mol_id}.gxl")
        graph = read_gxl_to_grakel(path)
        graphs.append(graph)
    return graphs


def load_labels(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['id', 'label'])
    return df['id'].tolist(), df['label'].tolist()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["active", "inactive"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["active", "inactive"],
                yticklabels=["active", "inactive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def main():
    base_path = "."
    gxl_dir = os.path.join(base_path, "gxl")

    train_ids, train_labels = load_labels(os.path.join(base_path, "train.tsv"))
    val_ids, val_labels = load_labels(os.path.join(base_path, "validation.tsv"))

    train_graphs = load_graphs(gxl_dir, train_ids)
    val_graphs = load_graphs(gxl_dir, val_ids)

    # Compute WL kernel
    print("Computing kernel matrix...")
    gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 5}, {"name": "subtree_wl"}], normalize=True)
    K_train = gk.fit_transform(train_graphs)
    K_test = gk.transform(val_graphs)

    # Train SVM
    clf = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
    clf.fit(K_train, train_labels)

    # Predict
    pred = clf.predict(K_test)

    # Evaluate
    acc = accuracy_score(val_labels, pred)
    print(f"Validation Accuracy: {acc:.4f}")
    plot_confusion_matrix(val_labels, pred)


if __name__ == "__main__":
    main()
