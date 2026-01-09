import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Configuração geral
# -----------------------------
st.set_page_config(page_title="Clientes/Compras — Online Shoppers", layout="wide")
st.title("📊 Clientes/Compras — Online Shoppers Intention")
st.write("Ferramenta simples para análise do comportamento de compradores online.")

# -----------------------------
# Upload de dados
# -----------------------------
uploaded_file = st.sidebar.file_uploader("Upload do CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Visão geral dos dados")
    st.write("Dimensões:", df.shape)
    st.dataframe(df.head(10), use_container_width=True)

    # -----------------------------
    # Preparação dos dados
    # -----------------------------
    coluna_alvo = st.sidebar.text_input("Coluna alvo (binária)", value="Revenue")

    if coluna_alvo in df.columns:
        df[coluna_alvo] = df[coluna_alvo].map({True: 1, False: 0}).fillna(df[coluna_alvo])
        features = [c for c in df.columns if c != coluna_alvo]

        X = df[features].select_dtypes(include=[np.number])
        y = df[coluna_alvo]

        # -----------------------------
        # Treino e avaliação
        # -----------------------------
        test_size = st.sidebar.slider("Proporção de teste", 0.1, 0.4, 0.2, 0.05)
        random_state = st.sidebar.number_input("Random state", value=42, step=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Resultados")
        st.write("Acurácia:", round(accuracy_score(y_test, y_pred), 3))

        # -----------------------------
        # Download das previsões
        # -----------------------------
        resultados = pd.DataFrame({
            "Real": y_test,
            "Previsto": y_pred
        })

        csv = resultados.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Baixar previsões em CSV",
            data=csv,
            file_name="previsoes.csv",
            mime="text/csv"
        )
    else:
        st.error(f"A coluna alvo '{coluna_alvo}' não está no dataset.")
else:
    st.info("Carregue um arquivo CSV para começar.")