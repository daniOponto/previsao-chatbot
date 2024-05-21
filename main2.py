import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Função para carregar os dados do Excel
def carregar_dados_excel(nome_arquivo):
    df = pd.read_excel(nome_arquivo)
    return df

# Função para fazer previsões da data de entrega
def prever_data_entrega(df):
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

    return pipeline

# Função para buscar detalhes do pedido
def buscar_pedido(codigo_cliente, df):
    order_data = df[df['Order ID'] == codigo_cliente]
    if order_data.empty:
        return 'Nenhum pedido encontrado para este código de cliente.'
    else:
        order = order_data.iloc[0]  # Acessar o primeiro pedido encontrado
        formatted_order = f"Seguem os dados do seu pedido:\n"
        formatted_order += f"Foi realizado em {order['Order Date']} na cidade de {order['City']} no estado de {order['State']}.\n"
        formatted_order += f"A categoria de entrega é {order['Ship Mode']} e seu produto {order['Category']} chegará em {order['Ship Date']}."
        return formatted_order

# Função para exibir o atendimento virtual
def atendimento_virtual():
    st.title("Bem-vindo ao Atendimento Virtual da Hold Logistica!")
    st.write("Olá, meu nome é Ingor! Como posso ajudá-lo hoje?")
    while True:
        opcao = st.selectbox("Escolha uma opção:", ('Informações sobre a empresa', 'Informações sobre o pedido', 'Sair do atendimento'))

        if opcao == 'Informações sobre a empresa':
            opcao_empresa = st.selectbox("O que deseja saber sobre nós?", ('Mais informações sobre a empresa', 'Serviços prestados', 'Como entrar em contato conosco'))
            st.write(empresa_respostas.get(opcao_empresa, 'Opção inválida.'))
        elif opcao == 'Informações sobre o pedido':
            codigo_cliente = st.text_input("Insira o código do seu pedido:")
            if codigo_cliente:
                st.write(buscar_pedido(codigo_cliente, df))
        elif opcao == 'Sair do atendimento':
            st.write("Até logo!")
            break

# Configurações iniciais
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

# Carregar dados
df = carregar_dados_excel('bancoDeDadosPI.xlsx')  # Ajuste o caminho do arquivo conforme necessário

# Inicializar o chatbot
empresa_respostas = {
    '1': 'Nossa empresa foi fundada em 2022 e desde então temos trabalhado para fornecer os melhores produtos e serviços aos nossos clientes.',
    '2': 'Somos uma empresa que garante que seu produto chegue em perfeito estado até você. Você pode encontrar mais informações em nosso site ou entrar em contato conosco para obter detalhes específicos.',
    '3': 'Para entrar em contato conosco, você pode nos ligar no número XXX-XXXX, enviar um e-mail para contato@empresa.com ou visitar nossa sede no endereço Rua ABC, nº 123.'
}

# Botão para alternar entre funcionalidades
opcao = st.radio("Escolha uma opção:", ('Previsão de Data de Entrega', 'Atendimento Virtual'))

if opcao == 'Previsão de Data de Entrega':
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
        pipeline = prever_data_entrega(df)
        y_pred = pipeline.predict(entrada)

        # Exibir a previsão
        st.markdown("## Resultado da Previsão")
        st.success(f'A previsão para a data de entrega é de {round(y_pred[0], 2)} dias.')

elif opcao == 'Atendimento Virtual':
    atendimento_virtual()
