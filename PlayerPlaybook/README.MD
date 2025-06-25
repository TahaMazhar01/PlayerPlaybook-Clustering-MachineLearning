

---

# ⚽ PlayerPlaybook: Performance Clustering with K-Means

A machine learning project that applies K-Means clustering on player attributes from the FIFA dataset to group similar player types. This helps identify performance-based patterns using Principal Component Analysis (PCA) for 2D visualization.

> 📁 GitHub by: [@TahaMazhar01](https://github.com/TahaMazhar01)

---

## 📊 Project Highlights

* ✅ Used FIFA dataset with selected performance attributes
* ✅ Preprocessing with `StandardScaler`
* ✅ Clustered players using **K-Means** with `k=4` Using elbow method to determine optimal k
* ✅ Visualized clusters using **PCA**
* ✅ Saved both elbow curve and final cluster plots

---

## 🧠 Features Used

We clustered players using the following 5 attributes:

* **Potential**
* **Finishing**
* **Standing Tackle**
* **Short Passing**
* **Dribbling**

These were selected for their relevance in assessing player capabilities across offensive and defensive roles.

---

## 🔁 Workflow Overview

1. **Data Preprocessing**:

   * Selected relevant features
   * Handled missing values
   * Applied standard scaling

2. **Clustering**:

   * Determined optimal `k` using Elbow Method (result: `k=4`)
   * Applied K-Means clustering

3. **Dimensionality Reduction**:

   * Used PCA to reduce features to 2D for visualization

4. **Visualization**:

   * Created elbow plot: `elbow_plot.png`
   * Created cluster scatter plot: `player_clusters.png`

5. **Output**:

   * Clustered data saved to `clustered_players.csv`

---

## 📎 Example Visualizations

* 🧩 **Elbow Method Plot**
  Helps determine the ideal number of clusters by observing inertia drop.

* 🌐 **PCA Cluster Plot**
  Visualizes clustered players in 2D space using principal components.

---

## 📁 Files in This Project

```bash
.
├── fifa_eda_stats.csv            # Input dataset
├── player_clusters.png           # Final cluster plot
├── elbow_plot.png                # Elbow curve for K
├── clustered_players.csv         # Output file with cluster labels
├── player_clustering.py          # Full project code
└── README.md                     # This file
```

---

## 🛠️ Libraries Used

```python
pandas
matplotlib
seaborn
sklearn (KMeans, PCA, StandardScaler)
```

---


---

## 🙌 Author

Built with 💡 by **[TahaMazhar01](https://github.com/TahaMazhar01)**
Feel free to fork, star ⭐ and contribute!

---
