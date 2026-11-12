"""
Project 3: Unsupervised Learning (Customer Segmentation)
Pipeline: SCALE -> COMPRESS (PCA) -> CLUSTER (K-Means) -> TRANSLATE (Personas)

Usage:
    python segmentation.py                         # uses Mall_Customers.csv if present, else synthetic demo data
    python segmentation.py --data my_data.csv      # your own dataset
    python segmentation.py --k 4                   # force K instead of auto-selecting
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT = "outputs"


# ----------------------------------------------------------------------------
# 0. Data loading
# ----------------------------------------------------------------------------
def make_demo_data(n=200):
    """Synthetic Mall-style data (used only if no CSV is found)."""
    rng = np.random.default_rng(RANDOM_STATE)
    spec = [  # age, income(k$), score, %female
        (41, 88, 17, 0.48), (33, 87, 82, 0.54),
        (25, 26, 79, 0.60), (45, 26, 21, 0.42),
    ]
    rows = []
    for age, inc, sc, f in spec:
        m = n // len(spec)
        rows.append(pd.DataFrame({
            "Gender": np.where(rng.random(m) < f, "Female", "Male"),
            "Age": rng.normal(age, 6, m).clip(18, 70).round(),
            "Annual Income (k$)": rng.normal(inc, 8, m).clip(10, 140).round(),
            "Spending Score (1-100)": rng.normal(sc, 8, m).clip(1, 99).round(),
        }))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "CustomerID", range(1, len(df) + 1))
    return df


def load_data(path):
    if path and os.path.exists(path):
        print(f"[data] Loaded {path}")
        return pd.read_csv(path)
    if os.path.exists("Mall_Customers.csv"):
        print("[data] Loaded Mall_Customers.csv")
        return pd.read_csv("Mall_Customers.csv")
    print("[data] No CSV found -> using synthetic demo data")
    return make_demo_data()


# ----------------------------------------------------------------------------
# 1. SCALE
# ----------------------------------------------------------------------------
def preprocess(df, include_cat=False):
    df = df.copy()
    id_cols = [c for c in df.columns if "id" in c.lower()]
    df = df.drop(columns=id_cols)
    df = df.dropna()
    raw = df.copy()  # keep human-readable copy for personas
    # Categoricals (e.g. Gender) are excluded from distance calculations by default:
    # a 0/1 column has huge scaled variance and would split clusters by gender.
    # They are still used later to PROFILE each cluster (e.g. % Female).
    if include_cat:
        X = pd.get_dummies(df, drop_first=True, dtype=float)
    else:
        X = df.select_dtypes("number")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return raw, X, X_scaled, scaler


# ----------------------------------------------------------------------------
# 2. COMPRESS (PCA)
# ----------------------------------------------------------------------------
def run_pca(X_scaled, n_components=3):
    full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
    cum = np.cumsum(full.explained_variance_ratio_)
    k95 = int(np.argmax(cum >= 0.95) + 1)
    print(f"[pca] Components needed for 95% variance: {k95} of {X_scaled.shape[1]}")

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cum) + 1), cum, marker="o")
    plt.axhline(0.95, color="orange", ls="--", label="95% threshold")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA: Cumulative Explained Variance")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{OUT}/01_pca_variance.png", dpi=150); plt.close()

    n = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print(f"[pca] Using {n} components, variance retained: "
          f"{pca.explained_variance_ratio_.sum():.1%}")
    return pca, X_pca


# ----------------------------------------------------------------------------
# 3. CLUSTER (Elbow + Silhouette)
# ----------------------------------------------------------------------------
def find_k(X_pca, k_max=10):
    ks = list(range(2, k_max + 1))
    wcss, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_pca)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_pca, km.labels_))

    # Elbow: point farthest from the line joining first & last WCSS point
    p1, p2 = np.array([ks[0], wcss[0]]), np.array([ks[-1], wcss[-1]])
    pts = np.column_stack([ks, wcss])
    v = p2 - p1
    w = pts - p1
    d = np.abs(v[0] * w[:, 1] - v[1] * w[:, 0]) / np.linalg.norm(v)
    k_elbow = ks[int(np.argmax(d))]
    k_sil = ks[int(np.argmax(sil))]
    print(f"[k] Elbow suggests K={k_elbow} | Silhouette suggests K={k_sil} "
          f"(score={max(sil):.3f})")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ks, wcss, marker="o")
    ax[0].axvline(k_elbow, color="red", ls="--", label=f"Elbow K={k_elbow}")
    ax[0].set(xlabel="Number of Clusters (K)", ylabel="WCSS", title="Elbow Method")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ks, sil, marker="o", color="green")
    ax[1].axvline(k_sil, color="red", ls="--", label=f"Best K={k_sil}")
    ax[1].set(xlabel="Number of Clusters (K)", ylabel="Silhouette Score",
              title="Silhouette Analysis")
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/02_elbow_silhouette.png", dpi=150); plt.close()

    pd.DataFrame({"K": ks, "WCSS": wcss, "Silhouette": sil}).to_csv(
        f"{OUT}/k_selection_metrics.csv", index=False)
    # Prefer silhouette; fall back to elbow if both agree or sil is tied
    return k_sil, k_elbow


def plot_clusters(X_pca, labels, km, pca):
    ev = pca.explained_variance_ratio_
    fig = plt.figure(figsize=(7, 5.5))
    if X_pca.shape[1] >= 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap="tab10", s=30)
        ax.scatter(*km.cluster_centers_[:, :3].T, c="black", marker="X", s=200)
        ax.set_zlabel(f"PC3 ({ev[2]:.0%})")
    else:
        ax = fig.add_subplot(111)
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=30)
        ax.scatter(*km.cluster_centers_[:, :2].T, c="black", marker="X", s=200)
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title("Customer Clusters in PCA Space")
    plt.tight_layout(); plt.savefig(f"{OUT}/03_clusters_pca.png", dpi=150); plt.close()


# ----------------------------------------------------------------------------
# 4. TRANSLATE (Personas)
# ----------------------------------------------------------------------------
def reverse_centroids(km, pca, scaler, columns):
    """PCA-space centroids -> scaled space -> original units."""
    scaled = pca.inverse_transform(km.cluster_centers_)
    original = scaler.inverse_transform(scaled)
    return pd.DataFrame(original, columns=columns).round(1)


def name_persona(row, med_inc, med_score):
    inc = [c for c in row.index if "income" in c.lower()]
    sc = [c for c in row.index if "spend" in c.lower() or "score" in c.lower()]
    if not inc or not sc:
        return "Segment"
    hi_inc, hi_sc = row[inc[0]] >= med_inc, row[sc[0]] >= med_score
    return {
        (True, True): ("High-Value Trendsetters",
                       "Exclusive perks, early access, experiential marketing."),
        (True, False): ("Affluent Conservatives",
                        "High-touch support, warranties, loyalty programs."),
        (False, True): ("Budget-Conscious Explorers",
                        "Influencer campaigns, flash sales, buy-now-pay-later."),
        (False, False): ("Conservative Minimizers",
                         "Clear price value, basic utility, minimal spend."),
    }[(hi_inc, hi_sc)]


def build_personas(raw, labels, km, pca, scaler, X_cols):
    raw = raw.copy()
    raw["Cluster"] = labels
    num = raw.select_dtypes("number").drop(columns="Cluster")
    prof = raw.groupby("Cluster")[num.columns].mean().round(1)
    prof["Size"] = raw.groupby("Cluster").size()
    prof["Share %"] = (prof["Size"] / len(raw) * 100).round(1)
    for c in raw.select_dtypes(exclude="number").columns:
        share = pd.crosstab(raw["Cluster"], raw[c], normalize="index") * 100
        for cat in share.columns:
            prof[f"% {cat}"] = share[cat].round(1)

    inc = [c for c in num.columns if "income" in c.lower()]
    sc = [c for c in num.columns if "spend" in c.lower() or "score" in c.lower()]
    med_inc = raw[inc[0]].median() if inc else 0
    med_sc = raw[sc[0]].median() if sc else 0
    names, actions = [], []
    for _, r in prof.iterrows():
        n, a = name_persona(r, med_inc, med_sc)
        names.append(n); actions.append(a)
    prof["Persona"], prof["Recommended Action"] = names, actions

    print("\n[personas] Cluster profiles (from actual customer means):")
    print(prof.to_string())

    rc = reverse_centroids(km, pca, scaler, X_cols)
    rc.index.name = "Cluster"
    rc.to_csv(f"{OUT}/centroids_inverse_transformed.csv")
    print("\n[personas] Centroids reverse-engineered via PCA + Scaler inverse:")
    print(rc.to_string())
    prof.to_csv(f"{OUT}/persona_summary.csv")
    return prof


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--k", type=int, default=None, help="force number of clusters")
    ap.add_argument("--include-categorical", action="store_true",
                    help="one-hot encode categorical columns into the clustering features")
    ap.add_argument("--components", type=int, default=3, help="PCA dims for clustering (2 or 3)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    df = load_data(a.data)
    print(f"[data] Shape: {df.shape}")
    raw, X, X_scaled, scaler = preprocess(df, a.include_categorical)
    pca, X_pca = run_pca(X_scaled, a.components)
    k_sil, k_elbow = find_k(X_pca)
    k = a.k or k_sil
    print(f"[k] Final K = {k}")

    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_pca)
    labels = km.labels_
    print(f"[model] Silhouette = {silhouette_score(X_pca, labels):.3f} | WCSS = {km.inertia_:.1f}")
    plot_clusters(X_pca, labels, km, pca)

    prof = build_personas(raw, labels, km, pca, scaler, list(X.columns))
    out = df.copy(); out["Cluster"] = labels
    out["Persona"] = out["Cluster"].map(prof["Persona"])
    out.to_csv(f"{OUT}/customers_with_clusters.csv", index=False)
    print(f"\n[done] Results saved in ./{OUT}/")


if __name__ == "__main__":
    main()
