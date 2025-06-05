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

# --- Funções Auxiliares Atualizadas ---

def descompactar_e_listar_csvs(arquivo_zip):
    """
    Descompacta um arquivo ZIP em memória e retorna um dicionário com os nomes
    dos arquivos CSV e seu conteúdo em string.
    """
    try:
        conteudos_csv = {}
        with zipfile.ZipFile(arquivo_zip, 'r') as z:
            arquivos_no_zip = z.namelist()
            nomes_arquivos_csv = [f for f in arquivos_no_zip if f.lower().endswith('.csv')]
            
            if not nomes_arquivos_csv:
                st.error("Nenhum arquivo .csv foi encontrado dentro do .zip.")
                return None
            
            for nome_arquivo in nomes_arquivos_csv:
                with z.open(nome_arquivo) as csv_file:
                    conteudo_csv_bytes = csv_file.read()
                    # Decodifica tentando UTF-8, mas com fallback para latin-1 (comum em CSVs brasileiros)
                    try:
                        conteudos_csv[nome_arquivo] = conteudo_csv_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        conteudos_csv[nome_arquivo] = conteudo_csv_bytes.decode('latin-1')
                        
            return conteudos_csv
            
    except Exception as e:
        st.error(f"Erro ao processar o arquivo ZIP: {e}")
        return None

# --- Interface e Lógica Principal do Streamlit ---

st.set_page_config(page_title="Agente Multi-CSV com Gemini", layout="wide")
st.title("🤖 Agente de IA para Análise de Múltiplos CSVs")
st.write("Faça o upload de um arquivo .zip. Se houver mais de um .csv, você poderá escolher qual analisar.")

# Inicializa o estado da sessão para guardar os dados e o DF selecionado
if 'conteudos_csv' not in st.session_state:
    st.session_state.conteudos_csv = None
if 'df_selecionado' not in st.session_state:
    st.session_state.df_selecionado = None
if 'nome_arquivo_zip' not in st.session_state:
    st.session_state.nome_arquivo_zip = ""

# 1. Upload do arquivo ZIP
uploaded_file = st.file_uploader("Escolha um arquivo .zip", type="zip")

if uploaded_file is not None:
    # Processa o arquivo apenas se for um novo upload
    if st.session_state.nome_arquivo_zip != uploaded_file.name:
        st.session_state.nome_arquivo_zip = uploaded_file.name
        st.session_state.df_selecionado = None # Reseta o dataframe ao carregar novo zip
        
        with st.spinner('Descompactando e procurando arquivos CSV...'):
            st.session_state.conteudos_csv = descompactar_e_listar_csvs(uploaded_file)

# 2. Seletor de arquivo CSV (só aparece se arquivos foram encontrados)
if st.session_state.conteudos_csv:
    lista_de_csvs = list(st.session_state.conteudos_csv.keys())
    
    # Se houver mais de um CSV, mostra o seletor. Se houver apenas um, seleciona-o automaticamente.
    if len(lista_de_csvs) > 1:
        csv_selecionado = st.selectbox(
            "Encontramos múltiplos arquivos CSV. Por favor, escolha qual deseja analisar:",
            options=lista_de_csvs
        )
    else:
        csv_selecionado = lista_de_csvs[0]
        st.info(f"Analisando o arquivo: **{csv_selecionado}**")

    # Carrega o DataFrame correspondente ao CSV selecionado
    try:
        df = pd.read_csv(StringIO(st.session_state.conteudos_csv[csv_selecionado]))
        st.session_state.df_selecionado = df
        st.success(f"Arquivo '{csv_selecionado}' carregado! O dataframe tem {df.shape[0]} linhas e {df.shape[1]} colunas.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Não foi possível carregar o arquivo '{csv_selecionado}' como um DataFrame. Erro: {e}")
        st.session_state.df_selecionado = None

# 3. Caixa de pergunta e interação com o agente (só aparece se um DF foi carregado)
if st.session_state.df_selecionado is not None:
    st.header("Faça sua pergunta sobre os dados")
    
    pergunta_usuario = st.text_input("Ex: Qual o valor máximo na coluna 'Preço'?")

    if st.button("Perguntar ao Agente"):
        if pergunta_usuario:
            with st.spinner('O agente Gemini está pensando... 🤔'):
                try:
                    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)
                    
                    agente = create_pandas_dataframe_agent(
                        llm,
                        st.session_state.df_selecionado,
                        verbose=True,
                        # Adiciona um prefixo para dar mais contexto ao agente
                        prefix="Você é um agente especialista em análise de dados. Responda à pergunta do usuário de forma clara e concisa com base no dataframe fornecido."
                    )
                    
                    resposta = agente.run(pergunta_usuario)
                    
                    st.success("Resposta do Agente:")
                    st.write(resposta)
                    
                except Exception as e:
                    st.error(f"Ocorreu um erro ao consultar o agente: {e}")
        else:
            st.warning("Por favor, digite uma pergunta.")