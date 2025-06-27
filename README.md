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
**① 벡터화**
```python
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df['processed'])
```

**② 모델 학습**
```python
lr = LogisticRegression(max_iter=1000, solver='liblinear')
grid = GridSearchCV(lr, {'C':[0.01,0.1,1,10]}, cv=5, scoring='f1')
grid.fit(X_tfidf, y)
best_model = grid.best_estimator_
```
하이퍼파라미터 `C` 최적값 및 F1 스코어를 출력하며, 최적 모델을 저장한다.

**③ 평가 및 시각화**
- 혼돈행렬(`confusion_matrix`), 분류 리포트(`classification_report`) 출력

- ROC 커브 및 AUC 계산

- WordCloud, 상위 N-그램 TF-IDF 막대 그래프 생성

- 결과 파일: `df.csv`, `vectorizer.pkl`, `model.pkl` 생성

# 3. LSTM 기반 딥러닝 버전
**① 토크나이즈 및 시퀀스 생성**
```python
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['processed'])
sequences = tokenizer.texts_to_sequences(df['processed'])
X_seq = pad_sequences(sequences, maxlen=100, padding='post')
y = df['sentiment'].map({'negative':0,'positive':1}).values
```

**② 모델 정의 및 학습**
```python
model = Sequential([
    Embedding(5000,128,input_length=100),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])
model.compile(optimizer=Adam(5e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_seq, y, epochs=10,
          batch_size=64, validation_split=0.2)
model.save('lstm_model.h5')
with open('tokenizer.json','w') as f:
    f.write(tokenizer.to_json())
```
양방향 LSTM과 드롭아웃으로 과적합을 방지하며, 모델 및 토크나이저를 저장한다.

**③ 평가**
- Log Loss, Accuracy, Precision, Recall, F1-score, AUC-ROC 산출

- 혼돈행렬, 분류 리포트 출력

# 4. Streamlit 대시보드(`app.pt`)
- 사이드바 컨트롤

  - Positive Threshold 슬라이더

  - Logistic Regression `C` 값 조정 슬라이더

- 메트릭 카드: Accuracy, Precision, Recall 표시

- 시각화: 혼돈행렬, ROC Curve, TF-IDF 상위 단어 바 차트, WordCloud

- 입력 텍스트 실시간 감성 예측 기능 (LR 및 LSTM 모두 지원)
