import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Função para calcular diversidade produtiva (número de culturas diferentes por município)
def calcular_diversidade(df, municipio_col, cultura_col):
    return df.groupby(municipio_col)[cultura_col].nunique().rename('diversidade_produtiva')

def generate_detailed_report(y_test, y_pred, target_names):
    """Gera relatório detalhado com todas as métricas"""
    
    # Métricas por classe
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    # Acurácia geral
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=" * 80)
    print("RELATÓRIO DETALHADO DE MÉTRICAS DO MODELO")
    print("=" * 80)
    
    print(f"\n ACURÁCIA GERAL: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n MÉTRICAS POR CLASSE:")
    print("-" * 60)
    for i, class_name in enumerate(target_names):
        print(f"\n  CLASSE: {class_name}")
        print(f"   • Precisão:  {precision[i]:.4f} ({precision[i]*100:.2f}%)")
        print(f"   • Recall:    {recall[i]:.4f} ({recall[i]*100:.2f}%)")
        print(f"   • F1-Score:  {f1[i]:.4f} ({f1[i]*100:.2f}%)")
        print(f"   • Suporte:   {support[i]} amostras")
    
    # Médias
    print("\n MÉDIAS:")
    print("-" * 30)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print(f"Macro Avg    - Precisão: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
    print(f"Weighted Avg - Precisão: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

def plot_confusion_matrix(y_test, y_pred, target_names):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusão - Classificação de Risco de Desperdício')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(clf, feature_names):
    """Plota importância das features"""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Importância das Variáveis no Modelo')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()
    
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)

def analyze_risk_distribution(df):
    """Analisa distribuição de risco por região e culturas"""
    print("\n ANÁLISE POR REGIÃO:")
    print("-" * 40)
    if 'Região' in df.columns:
        region_risk = df.groupby(['Região', 'RISCO_DESPERDICIO']).size().unstack(fill_value=0)
        region_risk_pct = region_risk.div(region_risk.sum(axis=1), axis=0) * 100
        print(region_risk_pct.round(2))
    
    print("\n ANÁLISE POR GRUPO DE CULTURA:")
    print("-" * 45)
    culture_risk = df.groupby(['GRUPO_CULTURA', 'RISCO_DESPERDICIO']).size().unstack(fill_value=0)
    culture_risk_pct = culture_risk.div(culture_risk.sum(axis=1), axis=0) * 100
    print(culture_risk_pct.round(2))

def main():
    # Carregar a base de dados
    file_path = 'd:/Users/Leonardo/Desktop/LovatConsorcios/vbp_2024.xlsx'
    df = pd.read_excel(file_path, skiprows=1)

    print("CLASSIFICADOR DE RISCO DE DESPERDÍCIO EM MUNICÍPIOS")
    print("=" * 60)
    print(f"Dataset carregado: {len(df)} registros")

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
    print("Calculando diversidade produtiva...")
    diversidade = calcular_diversidade(df, 'MUNICIPIO', 'CULTURA')
    df = df.merge(diversidade, left_on='MUNICIPIO', right_index=True)
    
    # Converter colunas numéricas
    numeric_cols = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remover linhas com valores nulos nas colunas principais
    initial_rows = len(df)
    df = df.dropna(subset=numeric_cols + ['diversidade_produtiva'])
    print(f"Removidos {initial_rows - len(df)} registros com dados faltantes")
    print(f"Dataset final: {len(df)} registros")

    # Estatísticas descritivas
    print("\nESTATÍSTICAS DESCRITIVAS:")
    print("-" * 40)
    print(df[numeric_cols + ['diversidade_produtiva']].describe().round(2))

    # Criar classificação de risco usando quantis
    print("\nCriando classificação de risco...")
    prod_q = df['PRODUCAO'].quantile([0.33, 0.66])
    area_q = df['AREA_PLANTADA'].quantile([0.33, 0.66])
    valor_q = df['VALOR_BRUTO'].quantile([0.33, 0.66])
    div_q = df['diversidade_produtiva'].quantile([0.33, 0.66])

    def classificar_risco(row):
        score = 0
        # Quanto menor a produção, maior o risco
        if row['PRODUCAO'] <= prod_q.iloc[0]: score += 1
        elif row['PRODUCAO'] >= prod_q.iloc[1]: score -= 1
        
        # Quanto menor a área plantada, maior o risco
        if row['AREA_PLANTADA'] <= area_q.iloc[0]: score += 1
        elif row['AREA_PLANTADA'] >= area_q.iloc[1]: score -= 1
        
        # Quanto menor o valor bruto, maior o risco
        if row['VALOR_BRUTO'] <= valor_q.iloc[0]: score += 1
        elif row['VALOR_BRUTO'] >= valor_q.iloc[1]: score -= 1
        
        # Quanto menor a diversidade, maior o risco
        if row['diversidade_produtiva'] <= div_q.iloc[0]: score += 1
        elif row['diversidade_produtiva'] >= div_q.iloc[1]: score -= 1
        
        if score >= 2:
            return 'ALTO'
        elif score <= -2:
            return 'BAIXO'
        else:
            return 'MEDIO'

    df['RISCO_DESPERDICIO'] = df.apply(classificar_risco, axis=1)

    # Distribuição das classes
    risk_dist = df['RISCO_DESPERDICIO'].value_counts()
    print(f"\nDISTRIBUIÇÃO DE CLASSES:")
    for risk_level, count in risk_dist.items():
        pct = (count / len(df)) * 100
        print(f"   {risk_level}: {count} ({pct:.1f}%)")

    # Preparar dados para o modelo
    features = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 'diversidade_produtiva', 'GRUPO_CULTURA']
    X = df[features].copy()
    y = df['RISCO_DESPERDICIO']

    # Codificar variável categórica
    print("\nPreparando dados para o modelo...")
    le = LabelEncoder()
    X['GRUPO_CULTURA'] = le.fit_transform(X['GRUPO_CULTURA'])

    # Normalizar variáveis numéricas
    scaler = StandardScaler()
    numeric_features = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 'diversidade_produtiva']
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Conjunto de treino: {len(X_train)} amostras")
    print(f"Conjunto de teste: {len(X_test)} amostras")

    # Treinar modelo Random Forest
    print("\nTreinando modelo Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_train, y_train)

    # Predições
    y_pred = clf.predict(X_test)

    # Gerar relatório detalhado
    target_names = sorted(y.unique())
    metrics = generate_detailed_report(y_test, y_pred, target_names)

    # Relatório tradicional do scikit-learn
    print("\nRELATÓRIO SCIKIT-LEARN:")
    print("-" * 40)
    print(classification_report(y_test, y_pred, digits=4))

    # Importância das variáveis
    print("\nIMPORTÂNCIA DAS VARIÁVEIS:")
    print("-" * 40)
    feature_importance = plot_feature_importance(clf, features)
    print(feature_importance.round(4))

    # Matriz de confusão
    print("\nGerando visualizações...")
    plot_confusion_matrix(y_test, y_pred, target_names)

    # Análise por região e cultura
    analyze_risk_distribution(df)

    # Salvar resultados
    print("\nSalvando resultados...")
    results_df = df[['MUNICIPIO', 'GRUPO_CULTURA', 'PRODUCAO', 'AREA_PLANTADA', 
                     'VALOR_BRUTO', 'diversidade_produtiva', 'RISCO_DESPERDICIO']]
    results_df.to_csv('classificacao_risco_municipios.csv', index=False)
    print("Resultados salvos em 'classificacao_risco_municipios.csv'")

    return clf, df, metrics

if __name__ == "__main__":
    classifier, data, metrics = main()
    print("\nAnálise concluída com sucesso!")