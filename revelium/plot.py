import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, List
from smartscan import Assignments

from io import BytesIO
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_clusters(ids: List[str], embeddings: List[np.ndarray], assignments: Assignments, method='tsne', random_state=42, output_path: Optional[str] = None):
    """
    Plots clusters from ClusterResult using 2D embeddings.

    Args:
        ids (List[str]): List of item IDs in the same order as embeddings.
        embeddings (List[np.ndarray]): List of embeddings (any dimension).
        cluster_result (ClusterResult): Result from IncrementalClusterer.
        method (str): Dimensionality reduction method: 'tsne' or 'pca'.
        random_state (int): Random seed for reproducibility.
    """
    embeddings_array = np.stack(embeddings, axis=0)
    
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reduced = TSNE(n_components=2, random_state=random_state).fit_transform(embeddings_array)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reduced = PCA(n_components=2, random_state=random_state).fit_transform(embeddings_array)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    # Get cluster IDs for each item
    cluster_ids = [assignments.get(i, "unassigned") for i in ids]

    # Assign a color to each cluster
    unique_clusters = list(set(cluster_ids))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cid: c for cid, c in zip(unique_clusters, colors)}

    # Plot each point
    plt.figure(figsize=(8, 6))
    for cid in unique_clusters:
        idxs = [i for i, c in enumerate(cluster_ids) if c == cid]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], color=color_map[cid], label=cid, s=50, edgecolor='k')

    plt.title("Prompt Clusters")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    if output_path:
        plt.savefig(output_path)


def plot_clusters_bytes(ids: List[str], embeddings: List[np.ndarray], assignments: Assignments, method='tsne', random_state=42) -> bytes:
    embeddings_array = np.stack(embeddings, axis=0)
    
    if method == 'tsne':
        reduced = TSNE(n_components=2, random_state=random_state).fit_transform(embeddings_array)
    elif method == 'pca':
        reduced = PCA(n_components=2, random_state=random_state).fit_transform(embeddings_array)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    cluster_ids = [assignments.get(i, "unassigned") for i in ids]
    unique_clusters = list(set(cluster_ids))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cid: c for cid, c in zip(unique_clusters, colors)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in unique_clusters:
        idxs = [i for i, c in enumerate(cluster_ids) if c == cid]
        ax.scatter(reduced[idxs, 0], reduced[idxs, 1], color=color_map[cid], label=cid, s=50, edgecolor='k')

    ax.set_title("Prompt Clusters")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    # Save to in-memory bytes buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
