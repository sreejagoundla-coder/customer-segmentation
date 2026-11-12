# Customer Segmentation using Unsupervised Learning

**Data Science – Project 3 | DecodeLabs Industrial Training Kit (Batch 2026)**

Segment retail customers into actionable business personas using **StandardScaler → PCA → K-Means**, with the optimal number of clusters proven by the **Elbow Method** and **Silhouette Score**.

---

## 🎯 Objective

Use distance-based algorithms to discover hidden groupings in unlabeled retail data and translate them into marketing personas.

## 🧱 Pipeline (IPO Architecture)

| Step | Phase | What happens | Why |
|------|-------|--------------|-----|
| 1 | **Scale** | `StandardScaler` (z = (x − μ) / σ) | Euclidean distance is dominated by large-magnitude features (income vs. purchases) |
| 2 | **Compress** | PCA to 2–3 components | Beats the curse of dimensionality; enables visualization |
| 3 | **Cluster** | K-Means, K chosen via Elbow + Silhouette | K-Means can't pick K on its own, so it must be justified |
| 4 | **Translate** | Inverse-transform centroids + profile clusters | PCA coordinates mean nothing to marketing teams |

## 📁 Project Structure

```
customer-segmentation/
├── segmentation.py        # full pipeline
├── requirements.txt
├── README.md
├── Mall_Customers.csv     # (you add this – see Dataset)
└── outputs/               # generated on run
    ├── 01_pca_variance.png
    ├── 02_elbow_silhouette.png
    ├── 03_clusters_pca.png
    ├── k_selection_metrics.csv
    ├── centroids_inverse_transformed.csv
    ├── persona_summary.csv
    └── customers_with_clusters.csv
```

## 📊 Dataset

Recommended: **Mall Customers dataset** (Kaggle) with columns
`CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)`.

Download it, save as `Mall_Customers.csv` in the project folder.
Any other retail CSV works too (`--data your_file.csv`), including datasets with 20+ columns.
If no CSV is found, the script runs on **synthetic demo data** so you can test it instantly.

## ⚙️ Setup & Run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python segmentation.py
```

Options:

| Flag | Description |
|------|-------------|
| `--data FILE.csv` | Use your own dataset |
| `--k 4` | Force K instead of auto-selecting |
| `--components 2` | Cluster in 2D PCA space (default 3) |
| `--include-categorical` | One-hot encode categorical columns into the clustering features |

## 🔬 Methodology Notes

- **ID columns** are dropped automatically.
- **Categorical columns (e.g. Gender)** are *excluded from clustering by default* and used only to profile clusters (e.g. % Female). A 0/1 column has large scaled variance and can split clusters by that column alone.
- **PCA:** the script plots cumulative explained variance and reports how many components reach the **95% rule**. For wide datasets (20+ columns) this typically means a handful of components; the clustering step then uses 2–3 for visualization. With small datasets (3–4 features) PCA retains nearly all variance.
- **Choosing K:** Elbow (point of maximum distance from the WCSS line) and Silhouette are both reported; the final K is the silhouette maximum unless `--k` is given. If they disagree, inspect both plots and pick the K that also makes business sense.
- **Reproducibility:** `random_state=42`, `n_init=10`.
- **Persona centroids** are reported two ways: reverse-engineered via `scaler.inverse_transform(pca.inverse_transform(centroid))` (matches the project brief), and as actual per-cluster means of the raw data (most accurate when PCA dropped variance).

## 🧑‍🤝‍🧑 Example Personas

Typical result on Mall Customers (K = 4–5; values vary by dataset):

| Persona | Profile | Recommended Action |
|---------|---------|--------------------|
| **High-Value Trendsetters** | Younger, high income, high spend | Exclusive perks, early access, experiential marketing |
| **Affluent Conservatives** | Middle-aged, high income, low spend | High-touch support, warranties, loyalty programs |
| **Budget-Conscious Explorers** | Young, low income, high spend | Influencer campaigns, flash sales, buy-now-pay-later |
| **Conservative Minimizers** | Older, low income, low spend | Clear price value, basic utility |

Persona names are auto-assigned by comparing each cluster's income and spending to the dataset median; rename them to suit your data.

## ⚠️ Limitations

- K-Means assumes roughly spherical, similar-sized clusters.
- Results depend on feature choice and scaling.
- Clusters describe patterns, not causes. Validate with A/B tests before spending marketing budget.

## 🚀 Ideas to Extend

- 3D interactive scatter (Plotly) of PCA components
- Compare with Gaussian Mixture or DBSCAN / Hierarchical clustering
- Add RFM features (Recency, Frequency, Monetary)
- Deploy as a Streamlit dashboard

## 🛠 Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib

---
*Built for DecodeLabs – Data Science Project 3.*
