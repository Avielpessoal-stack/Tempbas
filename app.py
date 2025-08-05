import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import unicodedata
from io import BytesIO
from PIL import Image

# --- Page Configuration and Logo ---
try:
    logo = Image.open("logo.jpg")
    st.set_page_config(
        page_title="EstimaTB",
        page_icon=logo,
        layout="centered",
        initial_sidebar_state="auto"
    )
except FileNotFoundError:
    st.set_page_config(
        page_title="EstimaTB",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="auto"
    )

# --- Authentication ---
def check_password():
    """Returns `True` if the user has the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] in st.secrets["passwords"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password.
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Digite o Código de Acesso", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Digite o Código de Acesso", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Código de acesso incorreto.")
        return False
    else:
        # Password correct.
        return True

# --- Helper Functions ---
def normalize_text(text, for_filename=False):
    if not isinstance(text, str): return text
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    if for_filename:
        return "".join(c for c in text if c.isalnum() or c in " _-").replace(" ", "_")
    else:
        return text.lower().replace(" ", "").replace("_", "")

def rename_columns(df):
    COLUMN_MAP = {
        'data': 'Data', 'tmin': 'Tmin', 'tminima': 'Tmin', 'tmín': 'Tmin',
        'tmax': 'Tmax', 'tmaxima': 'Tmax', 'tmáx': 'Tmax', 'nf': 'NF', 'nfolhas': 'NF'
    }
    df.rename(columns=lambda col: COLUMN_MAP.get(normalize_text(col), col), inplace=True)
    return df

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python', decimal=',')
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else: return None, "Formato de arquivo não suportado."
        return rename_columns(df), None
    except Exception as e: return None, f"Erro crítico ao ler o arquivo: {e}"

def validate_data(df):
    required_cols = ['Data', 'Tmin', 'Tmax', 'NF']
    errors = []
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Colunas obrigatórias não encontradas: {', '.join(missing_cols)}.")
        return df, errors
    try:
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        if df['Data'].isnull().any(): errors.append("Coluna 'Data' contém valores inválidos.")
    except Exception: errors.append("Erro ao converter a coluna 'Data'.")
    for col in ['Tmin', 'Tmax', 'NF']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    if not errors:
        if df[['Tmin', 'Tmax']].isnull().any().any():
            errors.append("Colunas 'Tmin' ou 'Tmax' contêm valores não-numéricos.")
        elif (df['Tmin'] > df['Tmax']).any():
            errors.append("'Tmin' maior que 'Tmax' em algumas linhas.")
        if not df['NF'].dropna().is_monotonic_increasing:
           errors.append("Valores em 'NF' (Número de Folhas) não estão sempre aumentando.")
    return df, errors

def perform_analysis(df, tb_min, tb_max, tb_step):
    df['Tmed'] = (df['Tmin'] + df['Tmax']) / 2
    pheno_df = df.dropna(subset=['NF']).copy()
    if len(pheno_df) < 3: return None, "São necessários pelo menos 3 dias com medição de 'NF'."
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

def create_excel_report(analysis_data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        analysis_data['meteor_sheet'].to_excel(writer, sheet_name='Dados Meteor. Periodo', index=False)
        analysis_data['nf_sheet'].to_excel(writer, sheet_name='NF e STa', index=False)
        analysis_data['qme_sheet'].to_excel(writer, sheet_name='QME', index=False)
        workbook, worksheet = writer.book, writer.sheets['QME']
        chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
        num_rows = len(analysis_data['qme_sheet'])
        chart.add_series({'name': 'QME vs Tb', 'categories': ['QME', 1, 0, num_rows, 0], 'values': ['QME', 1, 1, num_rows, 1]})
        chart.set_title({'name': 'QME vs. Temperatura Base'})
        chart.set_x_axis({'name': 'Temperatura Base (ºC)'})
        chart.set_y_axis({'name': 'Quadrado Médio do Erro (QME)'})
        worksheet.insert_chart('F2', chart)
    return output.getvalue()

# --- Main Application UI ---
if check_password():
    try:
        st.image("logo.jpg", width=100)
    except FileNotFoundError:
        pass
    
    st.title("EstimaTB")
    st.markdown("##### Estimativa da Temperatura Basal a partir de dados brutos de campo.")
    st.markdown("---")

    with st.expander("Como usar o EstimaTB?"):
        st.markdown("A simplicidade é a nossa força. O **EstimaTB** realiza análises complexas a partir de um único arquivo com dados mínimos. Forneça uma planilha com as colunas `Data`, `Tmin`, `Tmax` e `NF` (Número de Folhas), e deixe a ciência de dados conosco.")
    
    analysis_name = st.text_input("Nome da Análise (opcional, para nomear o arquivo final)")
    uploaded_file = st.file_uploader("Carregue seu arquivo de dados (CSV ou Excel)", type=['csv', 'xls', 'xlsx'])
    
    with st.expander("Opções Avançadas"):
        col1, col2, col3 = st.columns(3)
        tb_min, tb_max, tb_step = col1.number_input("Tb Mínima", value=0.0), col2.number_input("Tb Máxima", value=20.0), col3.number_input("Passo", value=0.5, min_value=0.1)

    if st.button("Analisar Dados", type="primary", disabled=(uploaded_file is None)):
        data, error = load_data(uploaded_file)
        if error: st.error(f"**Erro no Carregamento:** {error}")
        else:
            validated_data, errors = validate_data(data)
            if errors:
                st.warning("Foram encontrados problemas com os dados:")
                for e in errors: st.error(f"⚠️ {e}")
            else:
                with st.spinner("Analisando..."):
                    st.session_state.analysis_data, error = perform_analysis(validated_data, tb_min, tb_max, tb_step)
                    if error: st.error(f"**Erro na Análise:** {error}")
                    else: st.session_state.analysis_name = analysis_name

    if 'analysis_data' in st.session_state and st.session_state.analysis_data:
        st.markdown("---")
        results_title = f"Resultados para: \"{st.session_state.analysis_name}\"" if st.session_state.analysis_name else "Resultados da Análise"
        st.header(results_title)
        
        best = st.session_state.analysis_data['best']
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperatura Basal (Tb)", f"{best['Temperatura (ºC)']:.1f} °C"), col2.metric("Menor QME", f"{best['QME']:.4f}"), col3.metric("Coeficiente R²", f"{best['R2']:.3f}")
        st.latex(f"NF = {best['Coef_Angular']:.3f} \\times STa + {best['Intercepto']:.3f}")
        
        qme_df = st.session_state.analysis_data['qme_sheet']
        fig = go.Figure(go.Scatter(x=qme_df['Temperatura (ºC)'], y=qme_df['QME'], mode='lines+markers'))
        fig.update_layout(title="QME vs. Temperatura Base", xaxis_title="Temperatura Base (°C)", yaxis_title="Quadrado Médio do Erro (QME)")
        st.plotly_chart(fig, use_container_width=True)
        
        excel_report = create_excel_report(st.session_state.analysis_data)
        user_name = st.session_state.get('analysis_name', '')
        filename = f"{normalize_text(user_name, for_filename=True)}.xlsx" if user_name else "relatorio_do_pesquisador_sem_nome.xlsx"
        button_label = f"Baixar Relatório para \"{user_name}\"" if user_name else "Baixar Relatório Completo"
        
        st.download_button(f"📥 {button_label}", excel_report, filename, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
