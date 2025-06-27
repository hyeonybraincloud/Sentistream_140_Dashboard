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

# 2. TF-IDF + Logistic Regression 버전
① 벡터화
```python
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df['processed'])
```

② 모델 학습
```python
lr = LogisticRegression(max_iter=1000, solver='liblinear')
grid = GridSearchCV(lr, {'C':[0.01,0.1,1,10]}, cv=5, scoring='f1')
grid.fit(X_tfidf, y)
best_model = grid.best_estimator_
```
하이퍼파라미터 `C` 최적값 및 F1 스코어를 출력하며, 최적 모델을 저장한다.

③ 평가 및 시각화
- 혼돈행렬(`confusion_matrix`), 분류 리포트(`classification_report`) 출력

- ROC 커브 및 AUC 계산

- WordCloud, 상위 N-그램 TF-IDF 막대 그래프 생성

- 결과 파일: `df.csv`, `vectorizer.pkl`, `model.pkl` 생성
