# Sentistream140 Dashboard
**Sentiment140 데이터셋**을 활용한 배치 감성 분석 대시보드이다. 두 가지 파이프라인을 제공한다.

- TF-IDF + Logistic Regression 버전

- LSTM 기반 딥러닝 버전

# 1. 데이터 로드 및 전처리
① Kaggle Sentiment140 데이터셋(training.1600000.processed.noemoticon.csv)을 로드한다.

② 라벨 0 → negative, 4 → positive로 매핑한다.

③ preprocess(text) 함수 정의

- 소문자 변환

- URL 및 이모지 제거

- 알파벳·공백 외 문자 제거

- 공백 기준 토큰화 → 불용어 제거 → 표제어 추출

- 최종 처리된 토큰을 공백으로 결합하여 반환 
