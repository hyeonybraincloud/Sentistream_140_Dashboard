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

"""# 3. 벡터화"""

# Colab 환경에서 TF-IDF 벡터화

from sklearn.feature_extraction.text import TfidfVectorizer

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

    # TF-IDF 벡터라이저 설정 및 변환
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_tfidf = vectorizer.fit_transform(df['processed'])

    # 결과 확인
    print(f"TF-IDF 행렬 크기: {X_tfidf.shape}")

"""# 4. 모델 학습"""

# Colab 환경에서 로지스틱 회귀 분류기 학습 (GridSearchCV)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 이전 단계에서 X_tfidf와 df가 정의되었다고 가정
try:
    X_tfidf
    df
except NameError:
    print("⚠️ 오류: X_tfidf 또는 df가 정의되지 않았습니다. 먼저 vectorize 단계 셀을 실행하세요.")
else:
    # 레이블 인코딩: negative=0, positive=1
    y = df['sentiment'].map({'negative': 0, 'positive': 1})

    # 하이퍼파라미터 그리드 설정
    param_grid = {'C': [0.01, 0.1, 1, 10]}

    # 로지스틱 회귀 모델 및 GridSearchCV 설정
    lr = LogisticRegression(max_iter=1000, solver='liblinear')
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    # 모델 학습
    grid_search.fit(X_tfidf, y)

    # 최적 결과 출력
    print("Best C:", grid_search.best_params_['C'])
    print("Best F1 Score:", grid_search.best_score_)

    # 최적 모델 저장
    best_model = grid_search.best_estimator_

"""# 5. 평가 및 시각화"""

# Colab 환경에서 모델 평가: 혼돈행렬, classification_report, ROC 커브

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# 이전 단계에서 best_model, X_tfidf, df가 정의되었다고 가정
try:
    best_model
    X_tfidf
    df
except NameError:
    print("⚠️ 오류: best_model, X_tfidf 또는 df가 정의되지 않았습니다. train_model 단계를 먼저 실행하세요.")
else:
    # 레이블 인코딩
    y_true = df['sentiment'].map({'negative': 0, 'positive': 1})

    # 예측 및 예측 확률
    y_pred = best_model.predict(X_tfidf)
    y_prob = best_model.predict_proba(X_tfidf)[:, 1]

    # 혼돈행렬 출력
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()

    # 분류 리포트 출력
    report = classification_report(y_true, y_pred, target_names=['negative', 'positive'])
    print("Classification Report:")
    print(report)

    # ROC 커브 계산
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # ROC 커브 시각화
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.show()

import pandas as pd
import pickle

# 1) 훈련 데이터 저장
df.to_csv('df.csv', index=False)

# 2) 벡터라이저 저장
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# 3) 최적 모델 저장
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("✅ df.csv, vectorizer.pkl, model.pkl 파일이 생성되었습니다.")

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# cat > app.py << 'EOF'
# import streamlit as st
# import pandas as pd
# import pickle
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     confusion_matrix, roc_curve, auc, precision_score, recall_score
# )
# import plotly.express as px
# from wordcloud import WordCloud
# 
# # 1. NLTK 세팅 (최초 1회)
# nltk.download('stopwords')
# nltk.download('wordnet')
# 
# # 2. 전처리 함수 (간단 공백 토큰화)
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'http\\S+|www\\.\\S+', '', text)
#     text = re.sub(r'[^a-z\\s]', ' ', text)
#     tokens = text.split()
#     return ' '.join(lemmatizer.lemmatize(t) for t in tokens
#                     if t not in stop_words and len(t)>1)
# 
# # 3. 데이터·모델 로드
# df = pd.read_csv('df.csv')
# vectorizer = pickle.load(open('vectorizer.pkl','rb'))
# default_model = pickle.load(open('model.pkl','rb'))
# 
# # 4. 사이드바 컨트롤
# st.sidebar.header("⚙️ Settings")
# 
# # 4.1 임계값 슬라이더
# threshold = st.sidebar.slider(
#     "Positive Threshold",
#     min_value=0.0, max_value=1.0, value=0.5, step=0.01
# )
# 
# # 4.2 하이퍼파라미터 C 조정
# C_val = st.sidebar.select_slider(
#     "LogReg C (regularization)",
#     options=[0.01, 0.1, 1.0, 10.0], value=1.0
# )
# 
# # 5. 모델 재학습 (C 변경 시)
# @st.cache(allow_output_mutation=True, show_spinner=False)
# def train(C):
#     X = vectorizer.transform(df['text'].apply(preprocess))
#     y = df['sentiment'].map({'negative':0,'positive':1})
#     lr = LogisticRegression(C=C, max_iter=1000, solver='liblinear')
#     lr.fit(X, y)
#     return lr
# 
# model = train(C_val)
# 
# # 6. 전체 데이터 평가 지표
# X_all = vectorizer.transform(df['text'].apply(preprocess))
# y_true = df['sentiment'].map({'negative':0,'positive':1})
# y_prob = model.predict_proba(X_all)[:,1]
# y_pred = (y_prob >= threshold).astype(int)
# 
# acc = (y_pred==y_true).mean()
# prec = precision_score(y_true, y_pred)
# rec = recall_score(y_true, y_pred)
# 
# st.set_page_config(layout="wide")
# st.title("🎯 Batch Sentiment Analysis Dashboard")
# 
# col1, col2, col3 = st.columns(3)
# col1.metric("Accuracy", f"{acc:.3f}")
# col2.metric("Precision", f"{prec:.3f}")
# col3.metric("Recall", f"{rec:.3f}")
# 
# # 7. 혼돈행렬 & ROC (Plotly)
# cm = confusion_matrix(y_true, y_pred)
# cm_df = pd.DataFrame(cm, index=['Actual Neg','Actual Pos'], columns=['Pred Neg','Pred Pos'])
# fig_cm = px.imshow(
#     cm_df, text_auto=True, color_continuous_scale='Blues',
#     labels={'x':'Predicted','y':'Actual'}, title="Confusion Matrix"
# )
# fpr, tpr, _ = roc_curve(y_true, y_prob)
# roc_auc = auc(fpr, tpr)
# fig_roc = px.area(
#     x=fpr, y=tpr,
#     title=f"ROC Curve (AUC={roc_auc:.2f})",
#     labels={'x':'False Positive Rate','y':'True Positive Rate'},
#     line_shape='linear'
# )
# fig_roc.add_shape(
#     type='line', line=dict(dash='dash'),
#     x0=0, x1=1, y0=0, y1=1
# )
# 
# st.plotly_chart(fig_cm, use_container_width=True)
# st.plotly_chart(fig_roc, use_container_width=True)
# 
# # 8. 상위 단어 막대그래프
# def top_ngrams(sentiment, n=20):
#     corpus = df[df['sentiment']==sentiment]['text'].apply(preprocess)
#     vec = TfidfVectorizer(max_features=5000, ngram_range=(1,1))
#     X = vec.fit_transform(corpus)
#     sums = X.sum(axis=0).A1
#     terms = vec.get_feature_names_out()
#     top = sorted(zip(terms, sums), key=lambda x: x[1], reverse=True)[:n]
#     return pd.DataFrame(top, columns=['term','tfidf'])
# 
# bar_neg = top_ngrams('negative')
# bar_pos = top_ngrams('positive')
# 
# fig_bar_pos = px.bar(bar_pos, x='term', y='tfidf',
#                      title="Top Positive Words", labels={'tfidf':'TF-IDF'})
# fig_bar_neg = px.bar(bar_neg, x='term', y='tfidf',
#                      title="Top Negative Words", labels={'tfidf':'TF-IDF'})
# 
# st.plotly_chart(fig_bar_pos, use_container_width=True)
# st.plotly_chart(fig_bar_neg, use_container_width=True)
# 
# # 9. 워드클라우드 (크기/레이아웃 조정)
# wc_pos = WordCloud(
#     width=600, height=300, background_color='white'
# ).generate(' '.join(df[df['sentiment']=='positive']['text'].apply(preprocess)))
# wc_neg = WordCloud(
#     width=600, height=300, background_color='white'
# ).generate(' '.join(df[df['sentiment']=='negative']['text'].apply(preprocess)))
# 
# st.subheader("Positive Word Cloud")
# st.image(wc_pos.to_array())
# 
# st.subheader("Negative Word Cloud")
# st.image(wc_neg.to_array())
# EOF

"""# 6. Streamlit"""

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
