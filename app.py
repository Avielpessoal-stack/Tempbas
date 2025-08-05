import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import unicodedata
from io import BytesIO
from PIL import Image

# --- Page Configuration: Must be the first Streamlit command ---
try:
    logo = Image.open("logo.jpg")
    st.set_page_config(page_title="EstimaTB", page_icon=logo, layout="wide")
except FileNotFoundError:
    st.set_page_config(page_title="EstimaTB", page_icon="🌿", layout="wide")

# --- Custom CSS for Styling & Centering ---
CSS = """
<style>
    /* Center the logo */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    /* Button Colors */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #4CAF50; /* Verde Discreto */
        color: white;
        border-color: #4CAF50;
    }
    div[data-testid="stDownloadButton"] > button {
        background-color: #DAA520; /* Trigo Louro */
        color: white;
        border-color: #DAA520;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --- Authentication Function (Validated) ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if "password_attempted" not in st.session_state:
        st.session_state.password_attempted = False

    def password_entered():
        st.session_state.password_attempted = True
        if st.session_state.get("password") in st.secrets.get("passwords", []):
            st.session_state.password_correct = True
            if "password" in st.session_state: del st.session_state["password"]
        else:
            st.session_state.password_correct = False

    if not st.session_state.get("password_correct", False):
        st.text_input(
            "Digite o Código de Acesso para continuar",
            type="password",
            on_change=password_entered,
            key="password"
        )
        if st.session_state.password_attempted and not st.session_state.password_correct:
            st.error("😕 Código de acesso incorreto.")
        return False
    return True

# --- All other Helper Functions (Validated) ---
def normalize_text(text, for_filename=False):
    if not isinstance(text, str): return text
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    if for_filename:
        return "".join(c for c in text if c.isalnum() or c in " _-").replace(" ", "_")
    else:
        return text.lower().replace(" ", "").replace("_", "")

def rename_columns(df):
    COLUMN_MAP = {'data': 'Data', 'tmin': 'Tmin', 'tminima': 'Tmin', 'tmín': 'Tmin', 'tmax': 'Tmax', 'tmaxima': 'Tmax', 'tmáx': 'Tmax', 'nf': 'NF', 'nfolhas': 'NF', 'numerodefolhas': 'NF'}
    df.rename(columns=lambda col: COLUMN_MAP.get(normalize_text(col), col), inplace=True)
    return df

def load_and_validate_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python', decimal=',')
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else: return None, ["Formato de arquivo não suportado."], None
        
        df = rename_columns(df)
        required_cols, errors = ['Data', 'Tmin', 'Tmax', 'NF'], []
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Colunas obrigatórias não encontradas: {', '.join(missing_cols)}.")
            return None, errors, None

        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        if df['Data'].isnull().any(): errors.append("Coluna 'Data' contém valores inválidos.")
        for col in ['Tmin', 'Tmax', 'NF']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        if df['NF'].notna().sum() < 3: errors.append("A coluna 'NF' possui menos de 3 valores numéricos válidos.")
        
        if not errors:
            if df[['Tmin', 'Tmax']].isnull().any().any(): errors.append("Colunas 'Tmin' ou 'Tmax' contêm valores não-numéricos.")
            elif (df['Tmin'] > df['Tmax']).any(): errors.append("'Tmin' maior que 'Tmax' em algumas linhas.")
            nf_series = df['NF'].dropna()
            if not nf_series.is_monotonic_increasing: errors.append("Valores em 'NF' (Número de Folhas) não estão sempre aumentando.")
        
        return df, errors, df.head()
    except Exception as e: return None, [f"Erro crítico ao ler o arquivo: {e}"], None

@st.cache_data
def perform_analysis(df_input, tb_min, tb_max, tb_step):
    df = df_input.copy()
    df['Tmed'] = (df['Tmin'] + df['Tmax']) / 2
    pheno_df = df.dropna(subset=['NF']).copy()
    sta_details_df = df[['Data', 'Tmin', 'Tmax', 'Tmed']].copy()
    results = []
    base_temps = np.arange(tb_min, tb_max + tb_step, tb_step)
    
    for tb in base_temps:
        tb_col_name = f"STa (Tb={tb:.1f})"
        df['STd'] = df['Tmed'] - tb
        df.loc[df['STd'] < 0, 'STd'] = 0
        sta_details_df[tb_col_name] = df['STd'].cumsum()
        sta_for_regression = sta_details_df.loc[pheno_df.index, tb_col_name]
        X, y = sta_for_regression.values.reshape(-1, 1), pheno_df['NF'].values
        model = LinearRegression().fit(X, y)
        results.append({'Temperatura (ºC)': tb, 'QME': mean_squared_error(y, model.predict(X)), 'R2': model.score(X, y), 'Coef_Angular': model.coef_[0], 'Intercepto': model.intercept_})
        
    qme_df = pd.DataFrame(results)
    best_result = qme_df.loc[qme_df['QME'].idxmin()]
    nf_sta_df = sta_details_df.loc[pheno_df.index].copy()
    nf_sta_df.insert(1, 'NF', pheno_df['NF'])
    
    return {"best": best_result, "qme_sheet": qme_df, "meteor_sheet": sta_details_df, "nf_sheet": nf_sta_df}, None

@st.cache_data
def create_excel_report(analysis_data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook, date_format, header_format = writer.book, writer.book.add_format({'num_format': 'dd/mm/yyyy'}), writer.book.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
        df_meteor, df_nf, df_qme = analysis_data['meteor_sheet'], analysis_data['nf_sheet'], analysis_data['qme_sheet']
        
        df_meteor.to_excel(writer, sheet_name='Dados Meteor. Periodo', index=False, header=False, startrow=1)
        df_nf.to_excel(writer, sheet_name='NF e STa', index=False, header=False, startrow=1)
        df_qme.to_excel(writer, sheet_name='QME', index=False, header=False, startrow=1)
        
        ws_meteor, ws_nf, ws_qme = writer.sheets['Dados Meteor. Periodo'], writer.sheets['NF e STa'], writer.sheets['QME']
        
        ws_meteor.set_column('A:A', 12, date_format)
        ws_nf.set_column('A:A', 12, date_format)
        
        for ws, df in [(ws_meteor, df_meteor), (ws_nf, df_nf), (ws_qme, df_qme)]:
            for col_num, value in enumerate(df.columns.values): 
                ws.write(0, col_num, value, header_format)
        
        chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
        chart.add_series({'name': 'QME vs Tb', 'categories': ['QME', 1, 0, len(df_qme), 0], 'values': ['QME', 1, 1, len(df_qme), 1]})
        chart.set_title({'name': 'QME vs. Temperatura Base'})
        chart.set_x_axis({'name': 'Temperatura Base (ºC)'})
        chart.set_y_axis({'name': 'Quadrado Médio do Erro (QME)'})
        ws_qme.insert_chart('F2', chart)
        
        ws_ex, best_tb, ws_ex.df = workbook.add_worksheet('Exemplo de Calculo'), analysis_data['best']['Temperatura (ºC)'], analysis_data['meteor_sheet'][['Data', 'Tmin', 'Tmax', 'Tmed']].copy()
        headers = ['Data', 'Tmin', 'Tmax', 'Tmed', f'STd (Tb={best_tb:.1f})', f'STa (Tb={best_tb:.1f})']
        
        for col_num, value in enumerate(headers): 
            ws_ex.write(0, col_num, value, header_format)
            
        for row_num in range(1, len(ws_ex.df) + 1):
            ws_ex.write(row_num, 0, ws_ex.df['Data'].iloc[row_num-1], date_format)
            ws_ex.write(row_num, 1, ws_ex.df['Tmin'].iloc[row_num-1])
            ws_ex.write(row_num, 2, ws_ex.df['Tmax'].iloc[row_num-1])
            ws_ex.write(row_num, 3, ws_ex.df['Tmed'].iloc[row_num-1])
            ws_ex.write_formula(row_num, 4, f'=MAX(0, D{row_num+1} - {best_tb})')
            if row_num == 1: 
                ws_ex.write_formula(row_num, 5, f'=E{row_num+1}')
            else: 
                ws_ex.write_formula(row_num, 5, f'=F{row_num} + E{row_num+1}')
        ws_ex.set_column('A:A', 12)
        
    return output.getvalue()

# --- Main Application UI ---
if 'analysis_data' not in st.session_state: st.session_state.analysis_data = None
if 'analysis_error' not in st.session_state: st.session_state.analysis_error = None
if 'analysis_name' not in st.session_state: st.session_state.analysis_name = ""
if 'df_validated' not in st.session_state: st.session_state.df_validated = None
if 'validation_errors' not in st.session_state: st.session_state.validation_errors = []

if check_password():
    # --- Logo Display ---
    try:
        logo_image = Image.open("logo.jpg")
        width = 600
        w_percent = (width / float(logo_image.size[0]))
        h_size = int((float(logo_image.size[1]) * float(w_percent)))
        resized_logo = logo_image.resize((width, h_size), Image.Resampling.LANCZOS)
        
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image(resized_logo, output_format="JPEG")
        st.markdown('</div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.title("EstimaTB 🌿")

    # --- INSTRUÇÕES DE USO COMPLETAS ---
    with st.expander("Como usar o EstimaTB?"):
        st.markdown("""
        O **EstimaTB** foi desenhado para ser poderoso e simples. Siga os passos abaixo para obter a sua análise:

        **1. Prepare o seu ficheiro de dados:**
        * Pode ser um ficheiro `.csv` ou `.xlsx` (Excel).
        * O ficheiro deve conter **exatamente** 4 colunas com os seguintes cabeçalhos (ou variações próximas, como "Tminima", "tmín", etc.):
            * `Data`: As datas das observações (ex: 01/08/2024).
            * `Tmin`: As temperaturas mínimas diárias.
            * `Tmax`: As temperaturas máximas diárias.
            * `NF`: O número de folhas acumulado. **Importante:** deixe as células desta coluna em branco nos dias em que não houve medição de folhas.

        **2. Dê um nome à sua análise (opcional):**
        * Preencher este campo irá nomear o relatório em Excel que será gerado, facilitando a sua organização (ex: "Cultivar XPTO - Época 1").

        **3. Carregue o ficheiro:**
        * Clique no botão de upload e selecione o seu ficheiro de dados.
        * Uma pré-visualização dos dados carregados aparecerá para que possa confirmar se a leitura foi correta.

        **4. Analise:**
        * Clique no botão verde **"Analisar Dados"**.
        * A aplicação irá calcular a Temperatura Basal ótima e apresentar os resultados na tela.

        **5. Descarregue o Relatório Completo:**
        * Após a análise, clique no botão amarelo para descarregar um ficheiro Excel completo, com todas as tabelas, cálculos e gráficos detalhados para a sua documentação.
        """)
    
    analysis_name = st.text_input("Nome da Análise (opcional)")
    uploaded_file = st.file_uploader("Carregue o seu ficheiro de dados", type=['csv', 'xls', 'xlsx'], label_visibility="collapsed")
    
    if uploaded_file:
        df, errors, head_df = load_and_validate_data(uploaded_file)
        st.session_state.df_validated, st.session_state.validation_errors = df, errors
        if not errors and head_df is not None:
            with st.expander("Pré-visualização dos Dados Carregados", expanded=True): st.dataframe(head_df)
        elif errors:
            st.warning("Foram encontrados problemas com os seus dados:")
            for e in errors: st.error(f"⚠️ {e}")

    with st.expander("Opções Avançadas"):
        c1, c2, c3 = st.columns(3)
        tb_min, tb_max, tb_step = c1.number_input("Tb Mínima", value=0.0), c2.number_input("Tb Máxima", value=20.0), c3.number_input("Passo", value=0.5, min_value=0.1)

    if st.button("Analisar Dados", type="primary", disabled=(uploaded_file is None), use_container_width=True):
        if st.session_state.validation_errors:
            st.error("Corrija os erros nos dados (indicados acima) antes de analisar.")
        else:
            with st.spinner("A analisar..."):
                analysis_data, error_msg = perform_analysis(st.session_state.df_validated, tb_min, tb_max, tb_step)
                st.session_state.analysis_data, st.session_state.analysis_error = analysis_data, error_msg
                st.session_state.analysis_name = analysis_name
    
    # --- Results Section ---
    if st.session_state.analysis_data:
        st.markdown("---")
        results_title = f"Resultados para: \"{st.session_state.analysis_name}\"" if st.session_state.analysis_name else "Resultados da Análise"
        st.header(results_title)
        
        best = st.session_state.analysis_data['best']
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric("Temperatura Basal (Tb)", f"{best['Temperatura (ºC)']:.1f} °C")
            st.metric("Menor QME", f"{best['QME']:.4f}")
            st.metric("Coeficiente R²", f"{best['R2']:.3f}")
            st.markdown("**Equação do Modelo:**")
            st.latex(f"NF = {best['Coef_Angular']:.3f} \\times STa + {best['Intercepto']:.3f}")

        with res_col2:
            qme_df = st.session_state.analysis_data['qme_sheet']
            fig = go.Figure(go.Scatter(x=qme_df['Temperatura (ºC)'], y=qme_df['QME'], mode='lines+markers', line=dict(color='#DAA520')))
            fig.update_xaxes(dtick=1)
            fig.add_vline(x=best['Temperatura (ºC)'], line_width=2, line_dash="dash", line_color="#4CAF50")
            fig.update_layout(title_text="QME vs. Temperatura Base", xaxis_title="Temperatura Base (°C)", yaxis_title="Quadrado Médio do Erro (QME)")
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        excel_report = create_excel_report(st.session_state.analysis_data)
        user_name = st.session_state.get('analysis_name', '')
        filename = f"{normalize_text(user_name, for_filename=True)}.xlsx" if user_name else "relatorio_do_pesquisador_sem_nome.xlsx"
        button_label = f"Descarregar Relatório para \"{user_name}\"" if user_name else "Descarregar Relatório Completo"
        st.download_button(f"📥 {button_label}", excel_report, filename, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)

    elif st.session_state.analysis_error:
        st.error(f"**Erro na Análise:** {st.session_state.analysis_error}")
