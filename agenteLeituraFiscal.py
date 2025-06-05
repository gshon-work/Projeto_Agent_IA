import streamlit as st
import pandas as pd
import zipfile
import os
from io import StringIO
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Carrega as variáveis de ambiente
load_dotenv()

def descompactar_e_ler_csvs(arquivo_zip):
    """
    Descompacta um ZIP e lê todos os arquivos CSV encontrados, retornando uma lista de DataFrames.
    """
    try:
        dataframes = []
        nomes_arquivos = []
        with zipfile.ZipFile(arquivo_zip, 'r') as z:
            arquivos_no_zip = z.namelist()
            nomes_arquivos_csv = [f for f in arquivos_no_zip if f.lower().endswith('.csv')]
            
            if not nomes_arquivos_csv:
                st.error("Nenhum arquivo .csv foi encontrado dentro do .zip.")
                return None, None
            
            for nome_arquivo in nomes_arquivos_csv:
                with z.open(nome_arquivo) as csv_file:
                    conteudo_csv_bytes = csv_file.read()
                    try:
                        conteudo_str = conteudo_csv_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        conteudo_str = conteudo_csv_bytes.decode('latin-1')
                    
                    df = pd.read_csv(StringIO(conteudo_str))
                    dataframes.append(df)
                    nomes_arquivos.append(nome_arquivo)
                        
            return dataframes, nomes_arquivos
            
    except Exception as e:
        st.error(f"Erro ao processar o arquivo ZIP: {e}")
        return None, None

# --- Interface e Lógica Principal do Streamlit ---

st.set_page_config(page_title="Agente Multi-DataFrame com Gemini", layout="wide")
st.title("🤖 Agente de IA para Análise de Dados Consolidados")
st.write("Faça o upload de um .zip com múltiplos arquivos CSV. O agente analisará todos eles em conjunto.")

if 'lista_dfs' not in st.session_state:
    st.session_state.lista_dfs = None
if 'nomes_arquivos' not in st.session_state:
    st.session_state.nomes_arquivos = None
if 'nome_arquivo_zip' not in st.session_state:
    st.session_state.nome_arquivo_zip = ""

uploaded_file = st.file_uploader("Escolha um arquivo .zip", type="zip")

if uploaded_file is not None:
    if st.session_state.nome_arquivo_zip != uploaded_file.name:
        st.session_state.nome_arquivo_zip = uploaded_file.name
        
        with st.spinner('Descompactando e carregando todos os CSVs...'):
            dfs, nomes = descompactar_e_ler_csvs(uploaded_file)
            st.session_state.lista_dfs = dfs
            st.session_state.nomes_arquivos = nomes

if st.session_state.lista_dfs:
    st.success(f"{len(st.session_state.lista_dfs)} DataFrames carregados com sucesso!")
    
    with st.expander("Ver detalhes dos DataFrames carregados"):
        for nome, df in zip(st.session_state.nomes_arquivos, st.session_state.lista_dfs):
            st.write(f"**Arquivo: `{nome}`**")
            st.write(f"Shape: {df.shape[0]} linhas, {df.shape[1]} colunas")
            st.dataframe(df.head(3))
            st.divider()

if st.session_state.lista_dfs:
    st.header("Faça sua pergunta sobre os dados combinados")
    
    pergunta_usuario = st.text_input("Ex: Qual produto teve a maior quantidade total vendida?")

    if st.button("Perguntar ao Agente"):
        if pergunta_usuario:
            with st.spinner('O agente Gemini está analisando os múltiplos dataframes... 🤔'):
                try:
                 
                    llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
                    
                    agente = create_pandas_dataframe_agent(
                        llm,
                        st.session_state.lista_dfs, # <--- Passando a lista!
                        verbose=True,
                        # Adiciona um prefixo mais detalhado para ajudar o agente
                        prefix="""Você é um agente especialista em análise de dados com Pandas.
                        Você tem acesso a uma lista de DataFrames.
                        Quando uma pergunta for feita, você deve primeiro descobrir em qual DataFrame (ou em quais) está a informação.
                        Se precisar, você pode escrever e executar código para juntar ('merge') os DataFrames usando colunas em comum.
                        Responda à pergunta do usuário da forma objetiva e concisa.""",
                        allow_dangerous_code=True 
                    )
                    
                    resposta = agente.run(pergunta_usuario)
                    
                    st.success("Resposta do Agente:")
                    st.write(resposta)
                    
                except Exception as e:
                    st.error(f"Ocorreu um erro ao consultar o agente: {e}")
        else:
            st.warning("Por favor, digite uma pergunta.")