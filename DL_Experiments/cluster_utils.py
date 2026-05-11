import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

import plotly.express as px
import plotly.graph_objects as go


def get_X_2d(X_sample, sample_size=3000, random_state=42):
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    X_2d = tsne.fit_transform(X_sample)
    return X_2d


def plot_clusters_2d(X, cluster_labels, X_2d=None, title="Кластеры", sample_size=3000, random_state=42, add_legend=True):
    if X_2d is None:
        X_2d = get_X_2d(X, sample_size=sample_size)
    
    plt.figure(figsize=(12, 8))
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    sizes = np.array([(cluster_labels == i).sum() for i in unique_clusters])
    sizes_first_second = sorted(sizes, reverse=True)[:2]
    max_cluster = -1
    if sizes.shape[0] > 1:
        if sizes_first_second[0] / (sizes_first_second[1] + 1e-3) > 1:
            max_cluster = sizes.argmax()

    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    for i, cluster in enumerate(unique_clusters):
        if cluster < 0 or cluster == max_cluster:
            continue
        mask = cluster_labels == cluster
        points = X_2d[mask]
        
        if len(points) >= 4:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            polygon = Polygon(hull_points, alpha=0.2, color=colors[i])
            plt.gca().add_patch(polygon)
    
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1], 
            c=[colors[i]], label=f'Кластер {cluster}', s=10, alpha=0.7, edgecolors='none'
        )
    
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    if add_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_cluster_centroids_with_texts(X_sample, cluster_labels, texts, X_2d=None, vectorizer=None, n_top=3, sample_size=3000):

    if X_2d is None:
        X_2d = get_X_2d(X_sample, sample_size)

    labels_sample = cluster_labels
    texts_sample = texts
    indices = np.arange(X_sample.shape[0])
    
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_sample, cmap='tab20', s=15, alpha=0.6)
    plt.colorbar(scatter, label='Кластер')
    
    unique_clusters = np.unique(labels_sample)
    
    X_sample = np.asarray(X_sample)
    for cluster in unique_clusters:
        mask = labels_sample == cluster
        cluster_points = X_sample[mask]
        cluster_2d = X_2d[mask]
        
        if cluster_points.shape[0] == 0:
            continue
            
        centroid = cluster_points.mean(axis=0).reshape(1, -1)
        distances = cosine_distances(centroid, cluster_points)[0]
        closest_indices_in_sample = np.argsort(distances)[:min(n_top, len(distances))]
        
        for i, idx_in_sample in enumerate(closest_indices_in_sample):
            x, y = cluster_2d[idx_in_sample]
            
            circle = plt.Circle((x, y), radius=2, color='red', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            
            text_preview = texts_sample[mask][idx_in_sample][:50] + "..."
            plt.annotate(
                text_preview, 
                (x, y), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
            )
    
    plt.title('Кластеры с самыми типичными постами', fontsize=16)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.show()


def plot_clusters_3d(X, cluster_labels, sample_size=2000):
    if X.shape[0] > sample_size:
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
        labels_sample = cluster_labels[indices]
    else:
        X_sample = X
        labels_sample = cluster_labels
    
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()
    
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_sample)
    
    df_plot = pd.DataFrame({
        'PC1': X_3d[:, 0],
        'PC2': X_3d[:, 1],
        'PC3': X_3d[:, 2],
        'cluster': labels_sample.astype(str)
    })
    
    fig = px.scatter_3d(
        df_plot, x='PC1', y='PC2', z='PC3', color='cluster',
        title='3D визуализация кластеров (PCA)',
        opacity=0.7,
        width=800,
        height=600
    )
    fig.show()


def plot_cluster_channel_heatmap(cluster_data, channel_data):
    cross_tab = pd.crosstab(cluster_data, channel_data, normalize='columns')
    plt.figure(figsize=(14, 8))
    sns.heatmap(cross_tab, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Доля постов'})
    plt.title('Распределение каналов по кластерам', fontsize=16)
    plt.xlabel('Канал')
    plt.ylabel('Кластер')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return cross_tab


def get_X_3d(X_sample, sample_size=2000):
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_sample)
    return X_3d


def plot_clusters_3d_matplotlib(X_3d, labels_sample, sample_size=2000):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels_sample)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_sample == label
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            c=[colors[i]],
            label=f'Кластер {label}',
            s=5,
            alpha=0.6
        )
    
    ax.set_title('3D визуализация кластеров (PCA + matplotlib)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def draw_cluster_sizes(clusters):
    unique, counts = np.unique(clusters, return_counts=True)
    plt.bar(unique, counts)
    for i, (x, y) in enumerate(zip(unique, counts)):
        plt.text(x, y + max(counts)*0.01, str(y), ha='center', va='bottom', fontsize=9)
    plt.title('Размеры кластеров')
    plt.xlabel('Кластер')
    plt.ylabel('Количество постов')
    plt.show()



def get_top_texts_per_cluster(texts, cluster_labels, X_embedded, n_top=3):
    unique_clusters = np.unique(cluster_labels)
    result = {}

    if hasattr(X_embedded, 'toarray'):
        X_embedded = X_embedded.toarray()
    
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        cluster_texts = np.array(texts)[mask]
        cluster_points = X_embedded[mask]
        centroid = cluster_points.mean(axis=0)

        if hasattr(centroid, 'toarray'):
            centroid = centroid.toarray()
        centroid = centroid.reshape(1, -1)
        
        distances = cosine_distances(centroid, cluster_points)[0]
        closest_indices = np.argsort(distances)[:min(n_top, len(distances))]
        result[cluster] = [(cluster_texts[idx], distances[idx]) for idx in closest_indices]
    
    return result



def get_cluster_density(X_embedded, cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    densities = {}

    if hasattr(X_embedded, 'toarray'):
        X_embedded = X_embedded.toarray()
    
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        cluster_points = X_embedded[mask]
        
        if cluster_points.shape[0] <= 1:
            densities[cluster] = 0
            continue
        
        centroid = cluster_points.mean(axis=0)
        if hasattr(centroid, 'toarray'):
            centroid = centroid.toarray()
        
        centroid = centroid.reshape(1, -1)
        distances = cosine_distances(centroid, cluster_points)[0]
        
        avg_distance = distances.mean()
        densities[cluster] = avg_distance

    return dict(sorted(densities.items(), key=lambda x: x[1], reverse=False))

