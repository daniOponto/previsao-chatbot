import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Função para a aplicação de previsão
def previsao_entrega():
    # Carregar os dados
    df = pd.read_excel('bancoDeDadosPI.xlsx')

    # Convertendo as colunas de data para o tipo datetime
    df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
    df['ship_date'] = pd.to_datetime(df['ship_date'], dayfirst=True)

    # Diferença em dias entre as datas de pedido e entrega
    df['days_difference'] = (df['ship_date'] - df['order_date']).dt.days

    # Dados em conjunto de treinamento e teste
    X = df[['ship_mode', 'country', 'city', 'state', 'category']]
    y = df['days_difference']

    # Colunas que serão codificadas
    categorical_columns = ['ship_mode', 'country', 'city', 'state', 'category']

    # Codificar as colunas categóricas e treinar o modelo
    pipeline = Pipeline([
        ('encoder', ColumnTransformer([('encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')),
        ('model', KNeighborsRegressor(n_neighbors=5))
    ])

    # Treinando o modelo
    pipeline.fit(X, y)

    # Configurar cores do site
    st.markdown(
        """
        <style>
        .css-1aumxhk {
            color: white;
        }
        .css-1aumxhk a {
            color: #1a73e8;
        }
        .css-1rhbuit {
            color: #1a73e8;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Criar a interface gráfica
    st.title('Previsão de Data de Entrega')

    # Entrada do usuário
    st.markdown("## Informações de Entrada")
    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("País:", df['country'].unique())
        state = st.selectbox("Estado:", df['state'].unique())
        city = st.selectbox("Cidade:", df['city'].unique())
    with col2:
        ship_mode = st.selectbox("Modo de Envio:", df['ship_mode'].unique())
        category = st.selectbox("Categoria:", df['category'].unique())

    # Botão para fazer a previsão
    if st.button('Prever', key='prediction'):
        # DataFrame com os dados de entrada
        entrada = pd.DataFrame([[ship_mode, country, city, state, category]],
                               columns=['ship_mode', 'country', 'city', 'state', 'category'])

        # Previsão
        y_pred = pipeline.predict(entrada)

        # Exibir a previsão
        st.markdown("## Resultado da Previsão")
        st.success(f'A previsão para a data de entrega é de {round(y_pred[0], 2)} dias.')

# Função para a outra aplicação
def outra_aplicacao():
    st.write("Essa é a outra aplicação.")
    # Coloque aqui o código da outra aplicação

# Página principal
def main():
    st.sidebar.title('Menu')
    app_mode = st.sidebar.selectbox("Selecione a aplicação:", ["Previsão de Entrega", "Outra Aplicação"])

    if app_mode == "Previsão de Entrega":
        previsao_entrega()
    elif app_mode == "Outra Aplicação":
        outra_aplicacao()

if __name__ == "__main__":
    main()
