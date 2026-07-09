# Hierarchical-clustering-Credit-card-assignment

# Customer Segmentation of Credit Card Users Using Machine Learning

This project applies unsupervised machine learning — primarily **Hierarchical (Agglomerative) Clustering**, with **K-Means** as a comparison — to segment credit card customers based on their spending behavior. The goal is to uncover distinct customer groups from transaction patterns, purchase frequency, cash advance usage, and credit utilization, in order to support risk assessment and targeted marketing strategies.

## Notebook

`Hierarchical_Creditcard_Assignment_1.ipynb`

## Dataset

**File:** `CC GENERAL.csv` (not included — see [Setup](#setup))
**Size:** 8,950 customers × 18 columns

| Column | Description |
|---|---|
| CUST_ID | Credit card holder ID (categorical, dropped before modeling) |
| BALANCE | Balance remaining to make purchases |
| BALANCE_FREQUENCY | How often the balance is updated (0–1) |
| PURCHASES | Total amount of purchases |
| ONEOFF_PURCHASES | Maximum one-time purchase amount |
| INSTALLMENTS_PURCHASES | Amount purchased in installments |
| CASH_ADVANCE | Cash advance amount taken |
| PURCHASES_FREQUENCY | How often purchases are made |
| ONEOFF_PURCHASES_FREQUENCY | Frequency of one-off purchases |
| PURCHASES_INSTALLMENTS_FREQUENCY | Frequency of installment purchases |
| CASH_ADVANCE_FREQUENCY | How often cash advance is taken |
| CASH_ADVANCE_TRX | Number of cash advance transactions |
| PURCHASES_TRX | Number of purchase transactions |
| CREDIT_LIMIT | Credit card limit |
| PAYMENTS | Total payments made |
| MINIMUM_PAYMENTS | Minimum payments made |
| PRC_FULL_PAYMENT | Percent of full payments paid |
| TENURE | Number of months as a cardholder |

## Workflow

1. **Import Libraries** — pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
2. **Load Data** — read `CC GENERAL.csv`
3. **Explore & Visualize** — `.info()`, `.describe()`, null checks, correlation heatmap, boxplots for outliers
4. **Data Cleaning**
   - Drop `CUST_ID` (not useful for clustering)
   - Fill missing `CREDIT_LIMIT` with median
   - Fill missing `MINIMUM_PAYMENTS` with median
5. **Preprocessing** — scale all numeric features with `StandardScaler` (mean = 0, std = 1)
6. **Hierarchical Clustering**
   - Build a dendrogram (Ward linkage) to visualize how clusters merge and estimate an optimal cut
   - Compare linkage methods (`ward`, `complete`, `average`, `single`) using silhouette score
   - Determine optimal K using silhouette scores across K = 2–8
   - Fit final `AgglomerativeClustering` model (Ward linkage, K = 4)
7. **K-Means Clustering (comparison)**
   - Elbow method (inertia vs. K, K = 2–10)
   - Silhouette score validation across the same K range
   - Fit final `KMeans` model (k-means++, n_init = 20, K = 4)
8. **Dimensionality Reduction** — PCA to 2 components for 2D visualization of clusters and centroids
9. **Cluster Profiling** — compute mean feature values per cluster to characterize each segment
10. **Segment Naming & Business Interpretation**

| Cluster | Name | Approx. Size | Description |
|---|---|---|---|
| 0 | Low-Activity / Dormant | 3,977 (44.4%) | Low balance, low purchases, low cash advance |
| 1 | VIP High-Spenders | 409 (4.6%) | High purchases, high credit limit and payments |
| 2 | Cash-Advance Reliant | 1,197 (13.4%) | High cash advance usage relative to purchases |
| 3 | Responsible Regular Users | 3,367 (37.6%) | Moderate, healthy usage patterns |

11. **Business Recommendations** — tailored strategies per segment (e.g., premium rewards for VIPs, financial literacy programs for cash-advance-dependent users)
12. **Export Results** — labeled dataset saved to `CC_Clustered_Results.csv`

## Evaluation Metrics

- **Silhouette Score** — cluster cohesion/separation (higher is better)
- **Davies-Bouldin Score** — average similarity between clusters (lower is better)
- **Calinski-Harabasz Score** — variance ratio criterion
- **Inertia (WCSS)** — used for K-Means elbow method

Final K-Means model: K = 4, Inertia ≈ 99,062, Silhouette ≈ 0.2004

## Outputs

Running the notebook generates:
- `correlation_heatmap.png` — feature correlation heatmap
- `pca_clusters.png` — 2D PCA scatter plot of clusters
- `cluster_profiles.csv` — mean feature values per cluster
- `CC_Clustered_Results.csv` — original dataset with cluster labels and segment names

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Setup

1. Download the "Credit Card Dataset for Clustering" (`CC GENERAL.csv`), commonly available on Kaggle.
2. Update the file path in the data-loading cell to point to your local copy, e.g.:
   ```python
   df = pd.read_csv('CC GENERAL.csv')
   ```
   (The original notebook path points to a Google Drive location: `/content/drive/MyDrive/Colab Notebooks/.../CC GENERAL.csv` — update this if not running in Google Colab.)
3. Run all cells in order.

## Key Techniques Used

- Hierarchical Agglomerative Clustering (Ward, complete, average, single linkage)
- K-Means Clustering (k-means++ initialization)
- Dendrogram visualization
- Elbow Method
- Silhouette Score analysis
- Principal Component Analysis (PCA) for visualization
- StandardScaler feature scaling

## Notes

- The notebook contains some exploratory/duplicate cells (e.g., PCA and scaling performed more than once) as part of the iterative analysis process.
- A 600-sample subset is used for the dendrogram visualization since plotting all 8,950 points is too dense to interpret.
- `random_state=42` is used throughout for reproducibility.
