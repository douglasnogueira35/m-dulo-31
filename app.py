
import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

# -----------------------------
# Estilo visual (CSS)
# -----------------------------
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1, h2, h3 { color: #003366; }
    [data-testid="stSidebar"] { background-color: #f0f0f0; }
    a {
        background-color: #2e8b57;
        color: white !important;
        padding: 8px 16px;
        border-radius: 5px;
        text-decoration: none;
    }
    a:hover { background-color: #246b45; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Configuração geral
# -----------------------------
st.set_page_config(page_title="Clientes/Compras — Online Shoppers", layout="wide")
st.title("📊 Clientes/Compras — Online Shoppers Intention")
st.write("Ferramenta para análise e modelagem do comportamento de compradores online.")

# -----------------------------
# Sidebar — Upload e opções
# -----------------------------
st.sidebar.header("Configurações")
caminho_padrao = r"C:\Users\dougl\Downloads\online_shoppers_intention.csv"
uploaded_file = st.sidebar.file_uploader("Upload do CSV", type=["csv"])
usar_padrao = st.sidebar.checkbox("Usar arquivo padrão", value=True)

coluna_alvo = st.sidebar.text_input("Coluna alvo (binária)", value="Revenue")
modelo_escolhido = st.sidebar.selectbox("Modelo", ["Regressão Logística", "Random Forest"])
tamanho_teste = st.sidebar.slider("Proporção de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
cv_folds = st.sidebar.slider("Número de folds (CV)", 3, 10, 5, 1)

# -----------------------------
# Carregamento de dados
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif usar_padrao:
    df = pd.read_csv(caminho_padrao)
else:
    st.warning("Carregue um CSV ou selecione 'Usar arquivo padrão'.")
    st.stop()

st.subheader("Visão geral dos dados")
st.write("Dimensões:", df.shape)
st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# Preparação dos dados
# -----------------------------
st.subheader("Preparação dos dados")

if coluna_alvo not in df.columns:
    st.error(f"A coluna alvo '{coluna_alvo}' não está no dataset.")
    st.stop()

df[coluna_alvo] = df[coluna_alvo].map({True: 1, False: 0}).fillna(df[coluna_alvo])
if df[coluna_alvo].dtype == "object":
    df[coluna_alvo] = df[coluna_alvo].astype(str).str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    ).fillna(df[coluna_alvo])

df = df.dropna(subset=[coluna_alvo])

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

features = [c for c in df.columns if c != coluna_alvo]
num_features = [c for c in num_cols if c != coluna_alvo]
cat_features = [c for c in cat_cols if c != coluna_alvo]

st.write("Variáveis numéricas:", num_features)
st.write("Variáveis categóricas:", cat_features)

# -----------------------------
# Gráficos
# -----------------------------
st.subheader("Gráficos")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Distribuição da variável alvo")
    counts = df[coluna_alvo].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", color=["#2e8b57", "#ff6347"], ax=ax)
    ax.set_ylabel("Frequência")
    ax.set_xlabel("Classe")
    st.pyplot(fig)

with col2:
    if "Month" in df.columns:
        st.write("Distribuição por mês")
        fig, ax = plt.subplots()
        df["Month"].value_counts().plot(kind="bar", color="#4682b4", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Coluna 'Month' não encontrada.")

with col3:
    if "VisitorType" in df.columns:
        st.write("Tipos de visitantes")
        fig, ax = plt.subplots()
        df["VisitorType"].value_counts().plot(kind="bar", color="#ffa500", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Coluna 'VisitorType' não encontrada.")

st.write("Correlação entre variáveis numéricas")
if len(num_features) > 1:
    st.dataframe(df[num_features].corr().style.background_gradient(cmap="Blues"))
else:
    st.info("Poucas variáveis numéricas para correlação.")

# -----------------------------
# Treino e avaliação
# -----------------------------
st.subheader("Treino e avaliação do modelo")

X = df[features]
y = df[coluna_alvo]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=tamanho_teste, random_state=random_state, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

if modelo_escolhido == "Regressão Logística":
    model = LogisticRegression(max_iter=1000)
else:
    model = RandomForestClassifier(n_estimators=300, random_state=random_state)

pipe = Pipeline([("prep", preprocess), ("model", model)])

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
st.write(f"ROC-AUC médio (CV): {scores.mean():.3f}")

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

st.write("Métricas de teste:")
st.write({
    "Acurácia": round(accuracy_score(y_test, y_pred), 3),
    "Precisão": round(precision_score(y_test, y_pred), 3),
    "Recall": round(recall_score(y_test, y_pred), 3),
    "F1": round(f1_score(y_test, y_pred), 3),
    "ROC-AUC": round(roc_auc_score(y_test, y_proba), 3)
})

cm = confusion_matrix(y_test, y_pred)
st.write("Matriz de confusão")
st.dataframe(pd.DataFrame(cm, index=["Negativo", "Positivo"], columns=["Pred Neg", "Pred Pos"]))

# -----------------------------
# Downloads
# -----------------------------
st.subheader("Downloads")

pred_df = X_test.copy()
pred_df["Real"] = y_test.values
pred_df["Previsto"] = y_pred
pred_df["Probabilidade"] = y_proba

csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
b64 = base64.b64encode(csv_bytes).decode()
st.markdown(f'<a href="data:file/csv;base64,{b64}" download="predicoes_clientes_compras.csv">Baixar previsões (.csv)</a>', unsafe_allow_html=True)

model_bytes = pickle.dumps(pipe)
b64_model = base64.b64encode(model_bytes).decode()
st.markdown(f'<a href="data:file/octet-stream;base64,{b64_model}" download="modelo_clientes_compras.pkl">Baixar modelo treinado (.pkl)</a>', unsafe_allow_html=True)

st.success("Aplicação pronta! Carregue o arquivo, treine o modelo e baixe os resultados.")