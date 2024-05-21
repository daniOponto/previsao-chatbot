import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# Respostas para as opções sobre a empresa
empresa_respostas = {
    '1': 'Nossa empresa foi fundada em 2022 e desde então temos trabalhado para fornecer os melhores produtos e serviços aos nossos clientes.',
    '2': 'Somos uma empresa que garante que seu produto chegue em perfeito estado até você. Você pode encontrar mais informações em nosso site ou entrar em contato conosco para obter detalhes específicos.',
    '3': 'Para entrar em contato conosco, você pode nos ligar no número XXX-XXXX, enviar um e-mail para contato@empresa.com ou visitar nossa sede no endereço Rua ABC, nº 123.'
}

# Função Chatbot
def chatbot(df):  
    st.title("Bem-vindo ao Atendimento Virtual da Hold Logistica!")
    st.write("Olá, meu nome é Ingor! Como posso ajudá-lo hoje?")
    
    opcoes_menu = ('Informações sobre a empresa', 'Informações sobre o pedido', 'Sair do atendimento')

    # Criar widgets uma vez
    opcao = st.selectbox("Escolha uma opção:", opcoes_menu)

    if opcao == opcoes_menu[0]:  # Informações sobre a empresa
        opcao_empresa = st.selectbox("O que deseja saber sobre nós?", ('Mais informações sobre a empresa', 'Serviços prestados', 'Como entrar em contato conosco'))
        if opcao_empresa == 'Mais informações sobre a empresa':
            st.write(empresa_respostas['1'])
        elif opcao_empresa == 'Serviços prestados':
            st.write(empresa_respostas['2'])
        elif opcao_empresa == 'Como entrar em contato conosco':
            st.write(empresa_respostas['3'])
        else:
            st.write('Opção inválida.')
    elif opcao == opcoes_menu[1]:  # Informações sobre o pedido
        codigo_cliente = st.text_input("Insira o código do seu pedido:")
        if codigo_cliente:
            st.write(buscar_pedido(df, codigo_cliente))  
    elif opcao == opcoes_menu[2]:  # Sair do atendimento
        st.write("Até logo!")

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

# Botão para acessar o chatbot
if st.button('Acessar Chatbot'):
    chatbot(df)
