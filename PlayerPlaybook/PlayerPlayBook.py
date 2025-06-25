# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Data Loading ---
# Load dataset (assuming the CSV file is in the same directory as the script)
try:
    df = pd.read_csv("fifa_eda_stats.csv")
except FileNotFoundError:
    print("Error: 'fifa_eda_stats.csv' not found. Please make sure the file is in the same directory.")
    exit()

# --- Feature Selection and Scaling ---
# Select performance features
features = ['Potential', 'Finishing', 'StandingTackle', 'ShortPassing', 'Dribbling']
X = df[features].dropna() # Drop rows with missing values for these features

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-Means Clustering ---
# Elbow Method to confirm optimal k (Optional - for determining k)
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
sns.lineplot(x=list(k_range), y=inertia, marker='o', color='teal')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_plot.png") # Save the elbow plot
plt.show()

# Apply KMeans with k=4 (based on previous analysis)
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
# Fit KMeans on the scaled data
kmeans.fit(X_scaled)

# Predict the clusters and create a Series with the same index as X
clusters = pd.Series(kmeans.predict(X_scaled), index=X.index)

# Assign the clusters back to the original DataFrame using the index
df['Cluster'] = clusters

# --- PCA for 2D Visualization ---
pca = PCA(n_components=2)
# Fit and transform PCA on the scaled data
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for PCA results with the same index as X
X_pca_df = pd.DataFrame(X_pca, index=X.index, columns=['PC1', 'PC2'])

# Assign the PCA results back to the original DataFrame using the index
df['PC1'] = X_pca_df['PC1']
df['PC2'] = X_pca_df['PC2']

# --- Visualization ---
# Plot PCA clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df.dropna(subset=['Cluster', 'PC1', 'PC2']), x="PC1", y="PC2", hue="Cluster", palette="Set2", s=80)
plt.title("K-Means Clusters of Players (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("player_clusters.png") # Save the cluster plot
plt.show()

# --- Analyze Cluster Characteristics ---
cluster_characteristics = df.groupby('Cluster')[features].mean()
print("\nAverage characteristics of each cluster:")
display(cluster_characteristics)

# --- Save Results (Optional) ---
df.to_csv("clustered_players.csv", index=False)
print("\nClustered data saved to 'clustered_players.csv'")