# 📊 Clientes/Compras — Online Shoppers Intention

Ferramenta interativa desenvolvida em **Streamlit** para análise e modelagem do comportamento de compradores online, utilizando o dataset **Online Shoppers Intention**.  
O objetivo é identificar **quem mais compra, quem menos compra, clientes fiéis ou infiéis**, e gerar insights que podem apoiar estratégias de **marketing e fidelização**.

---

## 🚀 Funcionalidades
- Upload de arquivo CSV ou uso de caminho padrão.
- Visualização de dados com gráficos simples e coloridos.
- Treino de modelos de classificação (Regressão Logística e Random Forest).
- Avaliação com métricas: **Acurácia, Precisão, Recall, F1 e ROC-AUC**.
- Exibição da matriz de confusão para análise de desempenho.
- Downloads de previsões em CSV e do modelo treinado em PKL.
- Interface totalmente em português e com cores aplicadas para melhor estética.

---


---

## ⚙️ Instalação e execução local
1. Clone ou copie o projeto para sua máquina.  
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   http://localhost:8501
   📊 DatasetO dataset Online Shoppers Intention contém informações sobre sessões de navegação em um site de e-commerce, incluindo:- VisitorType: tipo de visitante (novo ou recorrente).
- Month: mês da visita.
- Weekend: se ocorreu no fim de semana.
- Revenue: variável alvo (se houve compra ou não).
- Outras variáveis de comportamento como tempo em páginas, taxas de saída e rejeição.
🎯 ObjetivoCom essa ferramenta, é possível:- Identificar clientes que mais compram e os que menos compram.
- Diferenciar clientes fiéis (recorrentes) dos infiéis (novos).
- Apoiar decisões de marketing, campanhas de fidelização e otimização da experiência do cliente.
