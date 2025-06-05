import streamlit as st
import pandas as pd
import zipfile
import os
from io import StringIO
from dotenv import load_dotenv

# ### MUDANÇA 1: Importar o LLM correto ###
# Remova ou comente a linha: from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI

# Importa o agente da LangChain (esta parte não muda)
from langchain_experimental.agents import create_pandas_dataframe_agent

# Carrega as variáveis de ambiente (agora vai procurar por GOOGLE_API_KEY)
load_dotenv()

# --- Funções Auxiliares (sem alterações aqui) ---

def descompactar_e_encontrar_csv(arquivo_zip):
    # ... (código da função exatamente como antes)
    try:
        with zipfile.ZipFile(arquivo_zip, 'r') as z:
            arquivos_no_zip = z.namelist()
            nome_arquivo_csv = next((f for f in arquivos_no_zip if f.lower().endswith('.csv')), None)
            if nome_arquivo_csv:
                with z.open(nome_arquivo_csv) as csv_file:
                    conteudo_csv_bytes = csv_file.read()
                    conteudo_csv_str = conteudo_csv_bytes.decode('utf-8', errors='ignore')
                    return conteudo_csv_str
            else:
                return None
    except Exception as e:
        st.error(f"Erro ao processar o arquivo ZIP: {e}")
        return None

# --- Interface do Streamlit (sem alterações aqui) ---

st.set_page_config(page_title="Agente de Análise de CSV com Gemini", layout="wide")
st.title("🤖 Agente de IA (com Gemini) para Análise de CSV")
st.write("Faça o upload de um arquivo .zip contendo um .csv e faça perguntas sobre os dados.")

# --- Lógica Principal ---

if 'df' not in st.session_state:
    st.session_state.df = None
if 'nome_arquivo' not in st.session_state:
    st.session_state.nome_arquivo = ""

uploaded_file = st.file_uploader("Escolha um arquivo .zip", type="zip")

if uploaded_file is not None:
    if st.session_state.nome_arquivo != uploaded_file.name:
        st.session_state.nome_arquivo = uploaded_file.name
        with st.spinner('Descompactando e carregando o CSV...'):
            conteudo_csv = descompactar_e_encontrar_csv(uploaded_file)
            if conteudo_csv:
                df = pd.read_csv(StringIO(conteudo_csv))
                st.session_state.df = df
                st.success(f"Arquivo CSV carregado! O dataframe tem {df.shape[0]} linhas e {df.shape[1]} colunas.")
                st.dataframe(df.head())
            else:
                st.error("Nenhum arquivo .csv foi encontrado dentro do .zip.")
                st.session_state.df = None

if st.session_state.df is not None:
    st.header("Faça sua pergunta sobre os dados")
    pergunta_usuario = st.text_input("Ex: Qual o valor médio da coluna 'vendas'?")

    if pergunta_usuario:
        with st.spinner('O agente Gemini está pensando... 🤔'):
            try:
                # ### MUDANÇA 2: Instanciar o LLM do Google (Gemini) ###
                # Em vez de OpenAI(...), usamos GoogleGenerativeAI(...)
                llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)

                # A criação do agente continua exatamente a mesma!
                # Ele apenas recebe o novo LLM que criamos.
                agente = create_pandas_dataframe_agent(llm, st.session_state.df, verbose=True)
                
                resposta = agente.run(pergunta_usuario)
                
                st.success("Resposta do Agente:")
                st.write(resposta)
                
            except Exception as e:
                st.error(f"Ocorreu um erro ao consultar o agente: {e}")