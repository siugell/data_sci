import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# 1) 데이터 불러오기
df = pd.read_csv('행정동별_월별_최종데이터.csv', encoding='utf-8')

# 2) 스케일링할 컬럼 지정
scale_cols = ['포함역개수', '하차승객수총합', '인구밀도', '면적', '인구']

# 3) Min–Max 스케일러 객체 생성 및 변환
scaler = RobustScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# 4) 결과 저장 (한글 깨짐 방지)
df.to_csv('행정동별_월별_최종데이터_Robust_scaled.csv', index=False, encoding='utf-8-sig')
