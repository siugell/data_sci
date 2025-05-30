import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

# 1) 스케일된 데이터 로드
df = pd.read_csv('행정동별_월별_최종데이터_Robust_scaled.csv', encoding='utf-8-sig')
scale_cols = ['포함역개수', '하차승객수총합', '인구밀도', '면적', '인구']
X = df[scale_cols].values

# 2) PCA로 차원 축소 (2차원)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_
print("PCA 설명 분산 비율:", explained)
print("누적 설명 분산 비율:", explained.cumsum())

# 3) 최적 파라미터 탐색
param_grid = {
    'n_clusters': range(2, 11),
    'init': ['k-means++', 'random'],
    'n_init': [10, 20],
    'max_iter': [100, 300, 500]
}

best_score = -1
best_params = None
results = []
for params in ParameterGrid(param_grid):
    km = KMeans(**params, random_state=42)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    results.append({**params, 'silhouette': score})
    if score > best_score:
        best_score, best_params = score, params

# 탐색 결과 저장
pd.DataFrame(results).to_csv('pca_kmeans_grid_results.csv', index=False, encoding='utf-8-sig')
print("최적 파라미터:", best_params)
print(f"최적 Silhouette Score: {best_score:.4f}")

# 4) 최적 파라미터로 클러스터링
km_best = KMeans(**best_params, random_state=42)
labels_best = km_best.fit_predict(X_pca)
df['cluster_pca'] = labels_best
df.to_csv('data_scaled_cluster_pca.csv', index=False, encoding='utf-8-sig')

# 5) 최적 모델 시각화
plt.figure(figsize=(8,6))
for cl in np.unique(labels_best):
    idx = labels_best == cl
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Cluster {cl}', s=20)
plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'PCA KMeans (k={best_params["n_clusters"]}, silhouette={best_score:.4f})')
plt.tight_layout()
plt.show()
