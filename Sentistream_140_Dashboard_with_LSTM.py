# 1. 데이터 로드
"""

# 1) Kaggle API 토큰 업로드
from google.colab import files
print("⚠️ kaggle.json 파일을 업로드해주세요.")
files.upload()  # 로컬의 kaggle.json 파일 선택

# 2) kaggle 설정 디렉토리 생성 및 권한 부여
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 3) Kaggle 패키지 설치
!pip install --upgrade kaggle

# 4) Sentiment140 데이터셋 다운로드 및 압축 해제
!kaggle datasets download -d kazanova/sentiment140 -p /content
!unzip -q /content/sentiment140.zip -d /content/sentiment140

# 5) pandas로 데이터셋 로드
import pandas as pd

df = pd.read_csv(
    '/content/sentiment140/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    names=['target', 'ids', 'date', 'flag', 'user', 'text']
)

# 라벨 매핑
df['sentiment'] = df['target'].map({0: 'negative', 4: 'positive'})
# 필요한 컬럼만 선택
df = df[['text', 'sentiment']]

# 데이터 확인
df.head()

"""# 2. 전처리 함수 작성"""

# Colab 환경용 수정된 preprocess 함수 (punkt 의존 제거)

# 필요한 패키지 설치 (이미 설치된 경우 생략)
!pip install --quiet nltk emoji

# NLTK 리소스 다운로드 (punkt 제거)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# 라이브러리 임포트
import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 불용어 및 표제어 도구 초기화
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """
    punkt 의존 없이:
    - 소문자 변환
    - URL, 이모지 제거
    - 특수문자 제거
    - 공백 기준 토큰화
    - 불용어 제거 및 표제어 추출
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = text.split()
    processed_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    return ' '.join(processed_tokens)

# 예시 테스트
example = "I love NLP! Visit https://example.com 😊 #AI #NLP"
print("Original:", example)
print("Processed:", preprocess(example))

"""# 3. Embedding 기반 토큰화 및 시퀀스 생성"""

from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense


# 이전 단계에서 df와 preprocess 함수가 정의되었다고 가정
try:
    df
    preprocess
except NameError:
    print("⚠️ 오류: df 또는 preprocess가 정의되지 않았습니다. 먼저 load_data와 preprocess 단계 셀을 실행하세요.")
else:
    # 전처리된 텍스트 컬럼이 없다면 생성
    if 'processed' not in df.columns:
        df['processed'] = df['text'].apply(preprocess)

    # --- 3. Embedding 기반 토큰화 & 시퀀스 생성 ---
    # 3-1) Tokenizer 설정 (상위 5,000개 단어만)
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['processed'])
    sequences = tokenizer.texts_to_sequences(df['processed'])

    # 3-2) 패딩 (최대 길이 100)
    X_seq = pad_sequences(sequences, maxlen=100, padding='post')

    # 3-3) 레이블 이진 인코딩 (예: 'negative'/'positive' → 0/1)
    y = df['sentiment'].map({'negative':0, 'positive':1}).values  # 컬럼명·매핑은 데이터프레임에 맞게 조정

"""# 4. LSTM 분류기 정의 및 학습"""

# --- 4. LSTM 분류기 정의 & 학습 ---
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Embedding(5000, 128, input_length=100),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=5e-4),  # 학습률 절반으로 낮춤
    loss='binary_crossentropy',
    metrics=['accuracy']
)

X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y, test_size=0.2, stratify=y, random_state=42
)

# 학습
history = model.fit(
    X_train, y_train,
    epochs=10,               # 에폭 수 늘려서 실험
    batch_size=64,
    validation_data=(X_val, y_val)
)

# ▼ 여기서부터 토크나이저 저장 ▼
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json','w', encoding='utf-8') as f:
    f.write(tokenizer_json)

"""# 5. LSTM 모델 평가 및 저장"""

# 5. LSTM 모델 평가 및 추가 지표 계산(sklearn)
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 5-1) 예측 확률 및 이진 예측값 생성
y_prob = model.predict(X_seq).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# 5-2) 성능 지표 계산
loss     = log_loss(y, y_prob)
accuracy = accuracy_score(y, y_pred)
prec   = precision_score(y, y_pred)
rec    = recall_score(y, y_pred)
f1     = f1_score(y, y_pred)
auc    = roc_auc_score(y, y_prob)
cm     = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

# 5-3) 결과 출력
print(f"LSTM 모델 — loss (log_loss): {loss:.4f}, accuracy: {accuracy:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

"""# 6. LSTM 모델 저장"""

# 6. LSTM 모델 저장
model.save('lstm_model.h5')
print("✅ LSTM 모델을 'lstm_model.h5'로 저장했습니다.")

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# cat > app.py << 'EOF'
# import streamlit as st
# import json
# import re
# import emoji
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# 
# # 1) 모델 & 토크나이저 로드
# model = load_model('lstm_model.h5')
# with open('tokenizer.json','r', encoding='utf-8') as f:
#     tokenizer = tokenizer_from_json(f.read())
# 
# # 2) 전처리 함수 (학습 때와 동일하게 복사)
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'http\\S+|www\\.\\S+', '', text)
#     text = emoji.replace_emoji(text, replace='')
#     text = re.sub(r'[^a-z\\s]', ' ', text)
#     tokens = text.split()
#     return ' '.join(lemmatizer.lemmatize(t) for t in tokens
#                     if t not in stop_words and len(t)>1)
# 
# # 3) Streamlit UI
# st.title("🎯 LSTM Sentiment Analysis")
# user_input = st.text_area("Enter text to analyze:")
# if st.button("Predict"):
#     processed = preprocess(user_input)
#     seq = tokenizer.texts_to_sequences([processed])
#     padded = pad_sequences(seq, maxlen=100, padding='post')
#     prob = model.predict(padded)[0,0]
#     st.write(f"Positive probability: {prob:.3f}")
# EOF

!pip install --quiet pyngrok
!ngrok authtoken (secret)

# 1) streamlit 설치 (필요 시)
!pip install --quiet streamlit

# 2) Streamlit을 0.0.0.0:8501에 바인딩하고 백그라운드로 실행
import os
os.system(
    "nohup streamlit run app.py "
    "--server.port=8501 "
    "--server.address=0.0.0.0 "
    "--server.headless=true > st.log 2>&1 &"
)

from pyngrok import ngrok

# 혹시 열린 터널이 있으면 닫고
ngrok.kill()

# 이제 0.0.0.0:8501 로 바인딩된 Streamlit을 터널링
public_url = ngrok.connect(addr="8501", proto="http")
print("🔗 Public URL:", public_url)
