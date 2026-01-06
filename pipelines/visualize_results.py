"""
visualize_results.py

用途:
 - 从 feature_extraction 的输出（固定路径 run_test_features_enhanced.csv）读取数据，生成可视化：
    * 模块置信热图 (module_meta)
    * 特征相关性与分布
    * 若存在标签: RandomForest 特征重要性柱状图
    * PCA + 聚类 (DBSCAN, KMeans) 可视化
    * 每簇的特征摘要 CSV
 - 输出文件夹: ./viz_outputs/
 - 直接运行示例:


说明:
 - 此版本已固定输入路径为:

 - 改进点:
   * 修复 dendrogram 的输入（对观测矩阵直接做 linkage，先 PCA 降维）
   * 自动估计 DBSCAN eps（k-distance percentile）并打印建议值
   * 移除近似恒定特征（VarianceThreshold）
   * 计算聚类质量指标（silhouette, calinski-harabasz）
   * 兼容没有 seaborn 的环境并设置中文字体回退
   * 所有保存图像使用 bbox_inches='tight'，dpi=200
"""
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Try to use seaborn style if seaborn is installed; otherwise fallback to matplotlib built-in
try:
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_context('talk')
except Exception:
    plt.style.use('ggplot')
    sns = None

# ensure chinese font (Windows)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import VarianceThreshold
from scipy.cluster.hierarchy import linkage, dendrogram

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_feature_file(path):
    df = pd.read_csv(path, encoding='utf-8')
    return df

def detect_module_meta_columns(df):
    module_cols = [c for c in df.columns if c.startswith('module_')]
    return module_cols

def top_feature_correlation(df, features, topn=40, out_dir='viz_outputs'):
    ensure_dir(out_dir)
    feats = [c for c in features if np.issubdtype(df[c].dtype, np.number)]
    if len(feats) == 0:
        print("[WARN] 没有可用的数值特征进行相关性分析")
        return []
    corr = df[feats].corr().abs()
    mean_corr = corr.mean().sort_values(ascending=False)
    top = mean_corr.index[:topn].tolist()
    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(df[top].corr(), cmap='vlag', center=0, annot=False)
    else:
        plt.imshow(df[top].corr(), cmap='bwr', aspect='auto')
    plt.title('Top features correlation heatmap', fontsize=14)
    plt.tight_layout()
    fp = os.path.join(out_dir, 'top_feature_correlation_heatmap.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print("[INFO] saved", fp)
    return top

def plot_module_meta_heatmap(df, module_cols, out_dir='viz_outputs'):
    ensure_dir(out_dir)
    if len(module_cols) == 0:
        print("[WARN] 无 module_meta 列（module_ 前缀），跳过模块置信热图")
        return
    mdf = df[module_cols].copy()
    maxrows = 200
    if mdf.shape[0] > maxrows:
        mdf = mdf.sample(maxrows, random_state=42)
    plt.figure(figsize=(12, max(4, mdf.shape[0]*0.06)))
    if sns is not None:
        sns.heatmap(mdf.T, cmap='YlGnBu', cbar_kws={'label': 'belief'}, vmin=0, vmax=1)
    else:
        plt.imshow(mdf.T, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
    plt.xlabel('sample index (subset)', fontsize=12)
    plt.ylabel('module', fontsize=12)
    plt.title('Module belief heatmap (rows: modules, cols: samples)', fontsize=14)
    plt.tight_layout()
    fp = os.path.join(out_dir, 'module_belief_heatmap.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print("[INFO] saved", fp)

def cluster_and_plot(df, feature_cols, out_dir='viz_outputs', n_clusters=4):
    ensure_dir(out_dir)
    if len(feature_cols) == 0:
        print("[WARN] 没有用于聚类的特征列")
        return df
    X = df[feature_cols].fillna(0.0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # remove near-constant features
    try:
        vt = VarianceThreshold(threshold=1e-6)
        Xv = vt.fit_transform(Xs)
        if Xv.shape[1] < Xs.shape[1]:
            removed = Xs.shape[1] - Xv.shape[1]
            print(f"[INFO] VarianceThreshold removed {removed} near-constant features")
            Xs = Xv
    except Exception as e:
        print("[WARN] VarianceThreshold error:", e)

    print(f"[INFO] clustering: samples={Xs.shape[0]}, features={Xs.shape[1]}")
    nan_ratio = np.isnan(Xs).mean()
    print(f"[INFO] Nan ratio in features: {nan_ratio:.4f}")
    variances = np.var(Xs, axis=0)
    print(f"[INFO] features variance: min={variances.min():.3e}, mean={variances.mean():.3e}, max={variances.max():.3e}")

    # PCA for visualization
    pca_vis = PCA(n_components=2, random_state=42)
    try:
        Xp = pca_vis.fit_transform(Xs)
    except Exception:
        Xp = Xs[:, :2] if Xs.shape[1] >= 2 else np.hstack([Xs, np.zeros((Xs.shape[0], max(0,2-Xs.shape[1])))])

    # KMeans clustering with silhouette-based suggestion for k
    best_k = n_clusters
    try:
        k_max = min(8, max(3, int(np.sqrt(Xs.shape[0]))))
        k_range = list(range(2, k_max+1))
        best_score = -1.0
        for k in k_range:
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
            if len(set(km_tmp.labels_)) > 1:
                try:
                    s = silhouette_score(Xs, km_tmp.labels_)
                except Exception:
                    s = -1.0
                if s > best_score:
                    best_score = s
                    best_k = k
        print(f"[INFO] KMeans silhouette best_k={best_k}, score={best_score:.4f}")
    except Exception as e:
        print("[WARN] KMeans silhouette search failed:", e)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(Xs)
    labels_km = km.labels_

    # DBSCAN: estimate eps via k-distance heuristic
    try:
        k_db = 5
        nbrs = NearestNeighbors(n_neighbors=k_db).fit(Xs)
        distances, _ = nbrs.kneighbors(Xs)
        k_distances = np.sort(distances[:, -1])
        eps_guess = float(np.percentile(k_distances, 90))
        print(f"[INFO] DBSCAN eps_guess (90th percentile) = {eps_guess:.4f}")
        db = DBSCAN(eps=eps_guess, min_samples=5).fit(Xs)
        labels_db = db.labels_
        print(f"[INFO] DBSCAN found clusters (unique labels incl -1): {np.unique(labels_db)}")
    except Exception as e:
        print("[WARN] DBSCAN estimation failed:", e)
        labels_db = np.array([-1]*Xs.shape[0])

    # plot PCA scatter by KMeans
    plt.figure(figsize=(8,6))
    palette = (sns.color_palette('tab10', best_k) if sns is not None else None)
    for lab in np.unique(labels_km):
        mask = labels_km == lab
        color = palette[int(lab)%len(palette)] if palette is not None else None
        plt.scatter(Xp[mask,0], Xp[mask,1], s=40, color=color, alpha=0.85, label=f'KMeans_{lab}')
    plt.title('PCA projection colored by KMeans', fontsize=14)
    plt.xlabel('PC1', fontsize=12); plt.ylabel('PC2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    fp = os.path.join(out_dir, 'pca_kmeans.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print("[INFO] saved", fp)

    # plot PCA scatter by DBSCAN
    plt.figure(figsize=(8,6))
    uniq = np.unique(labels_db)
    for lab in uniq:
        mask = labels_db == lab
        lab_name = f"DB_{lab}"
        if palette is not None:
            col = 'gray' if lab == -1 else palette[int(abs(lab))%len(palette)]
        else:
            col = None
        plt.scatter(Xp[mask,0], Xp[mask,1], s=40, color=col, alpha=0.85, label=lab_name)
    plt.title('PCA projection colored by DBSCAN', fontsize=14)
    plt.xlabel('PC1', fontsize=12); plt.ylabel('PC2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    fp = os.path.join(out_dir, 'pca_dbscan.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print("[INFO] saved", fp)

    # hierarchical dendrogram on a reduced representation (subset)
    try:
        subset_n = min(200, Xs.shape[0])
        if subset_n < 4:
            raise ValueError("Not enough samples for dendrogram")
        X_subset = Xs[:subset_n]
        n_comp = min(10, max(2, X_subset.shape[1]//2))
        pca_loc = PCA(n_components=n_comp, random_state=42)
        X_red = pca_loc.fit_transform(X_subset)
        Z = linkage(X_red, method='ward')
        plt.figure(figsize=(10, 4))
        dendrogram(Z, no_labels=True, color_threshold=None, truncate_mode='lastp', p=40)
        plt.title('Hierarchical clustering dendrogram (subset, PCA reduced)', fontsize=14)
        plt.tight_layout()
        fp = os.path.join(out_dir, 'dendrogram_subset.png')
        plt.savefig(fp, dpi=200, bbox_inches='tight')
        plt.close()
        print("[INFO] saved", fp)
    except Exception as e:
        print("[WARN] dendrogram failed:", e)

    # cluster quality metrics
    try:
        if len(set(labels_km)) > 1 and Xs.shape[0] > len(set(labels_km)):
            sil = silhouette_score(Xs, labels_km)
            ch = calinski_harabasz_score(Xs, labels_km)
            print(f"[INFO] KMeans silhouette={sil:.4f}, calinski-harabasz={ch:.4f}")
    except Exception as e:
        print("[WARN] silhouette/ch score failed:", e)

    # attach cluster labels back to df and produce cluster summary
    df_out = df.copy()
    df_out['cluster_km'] = labels_km
    df_out['cluster_db'] = labels_db
    summary = df_out.groupby('cluster_km')[feature_cols].mean().T
    summary.to_csv(os.path.join(out_dir, 'cluster_kmeans_feature_mean.csv'))
    print("[INFO] saved cluster summary csv ->", os.path.join(out_dir, 'cluster_kmeans_feature_mean.csv'))
    return df_out

def feature_importance_with_label(df, feature_cols, label_col, out_dir='viz_outputs', topn=30):
    ensure_dir(out_dir)
    if label_col not in df.columns:
        print(f"[WARN] label_col {label_col} not in df, skip features importance")
        return None
    X = df[feature_cols].fillna(0.0).values
    y = df[label_col].values
    if y.dtype.kind in {'U','O','S'}:
        y_enc, classes = pd.factorize(y)
    else:
        y_enc = y
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y_enc)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:topn]
    feat_sorted = [feature_cols[i] for i in idx]
    imps_sorted = importances[idx]
    plt.figure(figsize=(8, max(4, len(feat_sorted)*0.25)))
    if sns is not None:
        sns.barplot(x=imps_sorted, y=feat_sorted, palette='viridis')
    else:
        plt.barh(feat_sorted, imps_sorted)
    plt.xlabel('Importance', fontsize=12); plt.ylabel('Feature', fontsize=12)
    plt.title('RandomForest features importance (top {})'.format(len(feat_sorted)), fontsize=14)
    plt.tight_layout()
    fp = os.path.join(out_dir, 'feature_importance_rf.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print("[INFO] saved", fp)
    return pd.DataFrame({'features': feat_sorted, 'importance': imps_sorted})

# --------------------------
# Fixed-input entrypoint
# --------------------------
if __name__ == "__main__":
    # 固定输入文件（已按你要求指定）
    input_csv = r""
    prefix = "run_test"
    outdir = r""
    n_clusters = 4
    label_col = None  # 若有标签列可填写，例如 "true_module"

    ensure_dir(outdir)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"features file not found: {input_csv}")

    df = load_feature_file(input_csv)
    print(f"[INFO] Loaded features file: {input_csv}  rows={len(df)} cols={len(df.columns)}")
    module_cols = detect_module_meta_columns(df)
    print("[INFO] detected module columns:", module_cols)

    exclude_prefixes = ['module_', 'seq_index', 'timestamp', 'rep']
    candidate_features = [c for c in df.columns if c not in module_cols and not any(c.startswith(pref) for pref in exclude_prefixes)]
    numeric_features = [c for c in candidate_features if np.issubdtype(df[c].dtype, np.number)]
    print(f"[INFO] numeric features count: {len(numeric_features)}")

    top_feats = top_feature_correlation(df, numeric_features, topn=40, out_dir=outdir)
    plot_module_meta_heatmap(df, module_cols, out_dir=outdir)

    use_feats = top_feats if top_feats is not None and len(top_feats)>0 else numeric_features
    df_with_clusters = cluster_and_plot(df, use_feats, out_dir=outdir, n_clusters=n_clusters)

    if label_col:
        fi = feature_importance_with_label(df_with_clusters, use_feats, label_col, out_dir=outdir, topn=40)
        if fi is not None:
            fi.to_csv(os.path.join(outdir, 'feature_importances_supervised.csv'), index=False)
            print("[INFO] saved supervised features importances csv")

    df_with_clusters.to_csv(os.path.join(outdir, f"{prefix}_with_clusters.csv"), index=False, encoding='utf-8-sig')
    print("[INFO] saved augmented dataframe with clusters ->", os.path.join(outdir, f"{prefix}_with_clusters.csv"))