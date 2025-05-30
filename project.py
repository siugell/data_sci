
# -*- coding: utf-8 -*-
"""
wifi_recommendation_model_ensemble_rf_comparison.py

행정동별 월별 최종 데이터를 활용해
구 타깃 인코딩, 클러스터 통계, 이상치 제거,
그룹 단위 검증 분할 후
1) RandomForest 회귀 모델 성능 평가
2) XGBoost + LightGBM + RandomForest 스태킹 앙상블 평가
을 수행하는 단일 스크립트 예제
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    print(f"[INFO] 데이터 로드 완료: {df.shape}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = df.copy()
    df_proc['year_month'] = pd.to_datetime(df_proc['year_month'].astype(str), format='%Y%m')
    df_proc['month'] = df_proc['year_month'].dt.month
    df_proc['month_sin'] = np.sin(2 * np.pi * df_proc['month']/12)
    df_proc['month_cos'] = np.cos(2 * np.pi * df_proc['month']/12)

    te_map = df_proc.groupby('구')['와이파이수'].mean().to_dict()
    df_proc['구_te'] = df_proc['구'].map(te_map)
    df_proc.drop(columns=['구'], inplace=True)

    df_proc['역당_평균_하차승객수'] = (
    df_proc['하차승객수총합'] / df_proc['포함역개수'].replace(0,1)
    )
    df_proc['구_te×면적'] = df_proc['구_te'] * df_proc['면적']

    df_proc['cluster_orig'] = df_proc['cluster_pca']
    df_proc = pd.get_dummies(df_proc, columns=['cluster_pca'], prefix='cluster')
    return df_proc


def add_cluster_stats(df_proc: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
    stats = df_orig.groupby('cluster_pca').agg({
    '하차승객수총합': ['mean','std'],
    '인구밀도': ['mean','std']
    })
    stats.columns = [f"{c[0]}_{c[1]}" for c in stats.columns]
    stats = stats.reset_index()
    df = df_proc.merge(stats, left_on='cluster_orig', right_on='cluster_pca', how='left')
    df.drop(columns=['cluster_orig','cluster_pca'], inplace=True)
    return df


def remove_outliers_iqr(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.DataFrame:
    Q1, Q3 = df[col].quantile([0.25,0.75])
    IQR = Q3 - Q1
    mask = df[col].between(Q1 - k*IQR, Q3 + k*IQR)
    removed = len(df) - mask.sum()
    print(f"[INFO] {col}: 이상치 제거 {removed} rows")
    return df[mask]


def split_groupwise(df: pd.DataFrame, target: str='와이파이수', group: str='행정동'):
    X = df.drop(columns=[group, 'year_month', target])
    y = df[target]
    groups = df[group]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    X_full, X_test = X.iloc[tr], X.iloc[te]
    y_full, y_test = y.iloc[tr], y.iloc[te]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr2, va = next(gss2.split(X_full, y_full, groups.iloc[tr]))
    return X_full.iloc[tr2], X_full.iloc[va], X_test, y_full.iloc[tr2], y_full.iloc[va], y_test


def evaluate(y_true, y_pred, label: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}] MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")


def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    evaluate(y_test, pred, 'RandomForest')
    return rf


def train_models(X_train, y_train):
    xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', learning_rate=0.05,
    max_depth=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=10.0,
    n_estimators=500, random_state=42, eval_metric='mae'
    )
    xgb_model.fit(X_train, y_train, verbose=False)

    lgb_model = lgb.LGBMRegressor(
    learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=10.0,
    n_estimators=500, random_state=42
    )
    lgb_model.fit(X_train, y_train)

    return xgb_model, lgb_model


def stacking_predict(models, X_train, y_train, X_test):
    preds_train = np.column_stack([m.predict(X_train) for m in models])
    preds_test = np.column_stack([m.predict(X_test) for m in models])
    meta = Ridge(alpha=1.0)
    meta.fit(preds_train, y_train)
    return meta.predict(preds_test)


def main():
    df = load_data('data_scaled_cluster_pca_adjusted.csv')
    df_proc = preprocess(df)
    df_proc = add_cluster_stats(df_proc, df)
    for col in ['역당_평균_하차승객수','하차승객수총합','인구밀도','면적']:
        df_proc = remove_outliers_iqr(df_proc, col)
    X_train, X_val, X_test, y_train, y_val, y_test = split_groupwise(df_proc)

    # 1) RandomForest 성능
    rf = train_and_evaluate_rf(X_train, y_train, X_test, y_test)

    # 2) 스태킹 앙상블(XGB + LGB + RF)
    models = list(train_models(X_train, y_train)) + [rf]
    y_pred = stacking_predict(models, X_train, y_train, X_test)
    evaluate(y_test, y_pred, 'Stacking (XGB+LGB+RF)')

if __name__ == '__main__':
    main()