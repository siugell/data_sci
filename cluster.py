import os
# 병렬 탐색 시 한글 경로 문제 방지
os.environ['JOBLIB_TEMP_FOLDER'] = r'C:\temp\joblib_temp'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -----------------------------
# 1) 이미 스케일된 CSV 로드
# -----------------------------
df = pd.read_csv('행정동별_월별_최종데이터_scaled.csv', encoding='utf-8-sig')
scale_cols = ['포함역개수', '하차승객수총합', '인구밀도', '면적', '인구']
X = df[scale_cols].values   # numpy array로 변환

# -----------------------------
# 2) cv: train/test 모두 전체 데이터
# -----------------------------
n_samples = X.shape[0]
cv = [(np.arange(n_samples), np.arange(n_samples))]

# -----------------------------
# 3) 파라미터 분포 정의
# -----------------------------
param_dist = {
    'n_clusters': randint(2, 11),
    'init': ['k-means++', 'random'],
    'n_init': randint(10, 31),
    'max_iter': randint(200, 1001)
}

# -----------------------------
# 4) 실루엣 스코어러 정의
# -----------------------------
def silhouette_scorer(estimator, X, y=None):
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)

# -----------------------------
# 5) RandomizedSearchCV 구성
# -----------------------------
rs = RandomizedSearchCV(
    estimator=KMeans(random_state=42),
    param_distributions=param_dist,
    n_iter=30,
    scoring=silhouette_scorer,  # ← 직접 함수 전달
    cv=cv,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# 6) 하이퍼파라미터 탐색 실행
# -----------------------------
rs.fit(X)

# -----------------------------
# 7) 결과 저장
# -----------------------------
# 탐색 결과
pd.DataFrame(rs.cv_results_) \
  .to_csv('randomized_kmeans_results.csv', index=False, encoding='utf-8-sig')

# 최적 모델로 클러스터 레이블 추가
best_km = rs.best_estimator_
df['cluster'] = best_km.predict(X)
df.to_csv('data_with_randomized_kmeans.csv', index=False, encoding='utf-8-sig')

print("Best Params:", rs.best_params_)
print("Best Silhouette Score:", rs.best_score_)

# -----------------------------
# 8) PCA 2차원 시각화
# -----------------------------
X_pca = PCA(n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(8,6))
for cl in np.unique(df['cluster']):
    idx = df['cluster'] == cl
    plt.scatter(X_pca[idx,0], X_pca[idx,1], label=f'Cluster {cl}', s=20)

plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('RandomizedSearchCV KMeans Clustering Result')
plt.tight_layout()
plt.show()

score = silhouette_score(X, df['cluster'])
print(f"Silhouette Score: {score:.4f}")