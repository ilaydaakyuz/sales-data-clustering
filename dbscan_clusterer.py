import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

def optimize_eps_and_min_samples(X, eps_range=(0.5, 5.0), min_samples_range=(2, 10), eps_step=0.1):
    best_score = -1
    best_eps = None
    best_min_samples = None

    eps_values = np.arange(eps_range[0], eps_range[1] + eps_step, eps_step)

    for eps in eps_values:
        for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            if len(set(labels)) <= 1 or (set(labels) == {-1}):
                continue

            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples

    if best_eps is None or best_min_samples is None:
        best_eps = 1.5
        best_min_samples = 3

    return best_eps, best_min_samples

class DBSCANClusterer:
    def __init__(self, username, password, host, port, database):
        self.engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

    def fetch_data(self, query):
        df = pd.read_sql_query(query, self.engine)
        return df

    def find_optimal_eps(self, X_scaled, min_samples=3, save_path=None):
        neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        distances, _ = neighbors.kneighbors(X_scaled)
        distances = np.sort(distances[:, min_samples-1])

        kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
        optimal_eps = distances[kneedle.elbow]

        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps: {optimal_eps:.2f}')
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{min_samples}-th nearest neighbor distance')
        plt.title('Elbow Method for Optimal eps')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

        return optimal_eps

    def plot_clusters(self, df, x_feature, y_feature, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_feature], df[y_feature], c=df['cluster'], cmap='plasma', s=60)
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title('DBSCAN Clustering')
        plt.colorbar(label='Cluster No')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_scatter(self, X, labels, scatter_plot_path, x=None, y=None):
        plt.figure(figsize=(10, 8))
        
        if isinstance(X, pd.DataFrame):
            x_data = X[x]
            y_data = X[y]
        else:
            x_data = X[:, 0]
            y_data = X[:, 1]

        plt.scatter(x_data, y_data, c=labels, cmap='viridis', s=50)
        plt.title('Scatter Plot')
        plt.xlabel(x if x else 'Feature 1')
        plt.ylabel(y if y else 'Feature 2')
        plt.savefig(scatter_plot_path)
        plt.close()

    def preprocess_and_cluster(self, df, features, min_samples=3, elbow_plot_path=None, scatter_plot_path=None, scatter_x=None, scatter_y=None):
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        optimal_eps = self.find_optimal_eps(X_scaled, min_samples, save_path=elbow_plot_path)

        dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
        df["cluster"] = dbscan.fit_predict(X_scaled)

        if scatter_x and scatter_y and scatter_plot_path:
            self.plot_clusters(df, scatter_x, scatter_y, save_path=scatter_plot_path)

        outliers = df[df["cluster"] == -1]

        return df, outliers, optimal_eps
