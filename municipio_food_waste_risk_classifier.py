import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Função para calcular diversidade produtiva (número de culturas diferentes por município)
def calcular_diversidade(df, municipio_col, cultura_col):
    return df.groupby(municipio_col)[cultura_col].nunique().rename('diversidade_produtiva')

def main():
    # Carregar a base de dados
    file_path = 'd:/Users/Leonardo/Desktop/LovatConsorcios/vbp_2024.xlsx'
    df = pd.read_excel(file_path, skiprows=1)

    # Limpar nomes das colunas
    df.columns = [col.strip().replace('\n', '') for col in df.columns]
    
    # Mapear colunas do arquivo real
    column_mapping = {
        'Município': 'MUNICIPIO',
        'Produção': 'PRODUCAO', 
        'Área (ha)': 'AREA_PLANTADA',
        'VBP': 'VALOR_BRUTO',
        'Grupo': 'GRUPO_CULTURA',
        'Cultura': 'CULTURA'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Verificar se as colunas necessárias existem
    required_cols = ['MUNICIPIO', 'PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 'GRUPO_CULTURA', 'CULTURA']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Colunas disponíveis: {list(df.columns)}")
        raise Exception(f'Colunas não encontradas: {missing_cols}')

    # Calcular diversidade produtiva
    diversidade = calcular_diversidade(df, 'MUNICIPIO', 'CULTURA')
    df = df.merge(diversidade, left_on='MUNICIPIO', right_index=True)
    
    # Converter colunas numéricas
    numeric_cols = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remover linhas com valores nulos nas colunas principais
    df = df.dropna(subset=numeric_cols + ['diversidade_produtiva'])

    # Exemplo de criação de variável alvo (ajuste conforme contexto real):
    # Risco alto: produção baixa, área plantada baixa, valor bruto baixo, baixa diversidade
    # Risco baixo: produção alta, área plantada alta, valor bruto alto, alta diversidade
    # Risco médio: intermediário
    # Aqui, usamos quantis para criar a classificação
    prod_q = df['PRODUCAO'].quantile([0.33, 0.66])
    area_q = df['AREA_PLANTADA'].quantile([0.33, 0.66])
    valor_q = df['VALOR_BRUTO'].quantile([0.33, 0.66])
    div_q = df['diversidade_produtiva'].quantile([0.33, 0.66])

    def classificar_risco(row):
        score = 0
        if row['PRODUCAO'] <= prod_q.iloc[0]: score += 1
        elif row['PRODUCAO'] >= prod_q.iloc[1]: score -= 1
        if row['AREA_PLANTADA'] <= area_q.iloc[0]: score += 1
        elif row['AREA_PLANTADA'] >= area_q.iloc[1]: score -= 1
        if row['VALOR_BRUTO'] <= valor_q.iloc[0]: score += 1
        elif row['VALOR_BRUTO'] >= valor_q.iloc[1]: score -= 1
        if row['diversidade_produtiva'] <= div_q.iloc[0]: score += 1
        elif row['diversidade_produtiva'] >= div_q.iloc[1]: score -= 1
        if score >= 2:
            return 'ALTO'
        elif score <= -2:
            return 'BAIXO'
        else:
            return 'MEDIO'

    df['RISCO_DESPERDICIO'] = df.apply(classificar_risco, axis=1)

    # Features para o modelo
    features = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 'diversidade_produtiva', 'GRUPO_CULTURA']
    X = df[features].copy()
    y = df['RISCO_DESPERDICIO']

    # Codificar variável categórica
    if 'GRUPO_CULTURA' in X.columns:
        le = LabelEncoder()
        X['GRUPO_CULTURA'] = le.fit_transform(X['GRUPO_CULTURA'])

    # Normalizar variáveis numéricas
    scaler = StandardScaler()
    X[['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 'diversidade_produtiva']] = scaler.fit_transform(
        X[['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 'diversidade_produtiva']]
    )

    # Separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Treinar modelo
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predições
    y_pred = clf.predict(X_test)

    # Relatório de métricas
    print('Acurácia:', accuracy_score(y_test, y_pred))
    print('\nRelatório de Classificação:')
    print(classification_report(y_test, y_pred, digits=3))

    # Exemplo de importância das features
    importances = pd.Series(clf.feature_importances_, index=features)
    print('\nImportância das variáveis:')
    print(importances.sort_values(ascending=False))

if __name__ == "__main__":
    main()
