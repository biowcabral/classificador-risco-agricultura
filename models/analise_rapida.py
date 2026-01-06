"""
Sistema Avançado de Análise Temporal de Risco de Desperdício Agrícola
Versão OTIMIZADA para Datasets Grandes
Comparação Abrangente de Metodologias de Machine Learning
Autor: Sistema de Classificação de Risco Agrícola
Data: Janeiro 2026
"""

# =============================================================================
# IMPORTAÇÕES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob
import os
import re
import json
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Importações para Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, 
                            ConfusionMatrixDisplay)

# CLASSIFICAÇÃO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                             GradientBoostingClassifier, BaggingClassifier,
                             VotingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# CLUSTERING
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# FEATURE SELECTION (apenas os rápidos)
from sklearn.feature_selection import SelectKBest, f_classif

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RESULTADOS = {
    'models': {},
    'feature_selection': {},
    'clustering': {},
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

print("=" * 100)
print(" " * 15 + "SISTEMA AVANÇADO DE ANÁLISE TEMPORAL DE RISCO AGRÍCOLA")
print(" " * 20 + "Versão OTIMIZADA - Comparação de Metodologias de ML")
print("=" * 100)

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def padronizar_grupo_existente(grupo):
    """Padroniza nomes de grupos existentes nos dados"""
    if pd.isna(grupo):
        return 'Outros'
    
    grupo_str = str(grupo).strip().lower()
    grupo_str = grupo_str.replace('ã', 'a').replace('õ', 'o').replace('ç', 'c')
    grupo_str = grupo_str.replace('á', 'a').replace('é', 'e').replace('í', 'i')
    grupo_str = grupo_str.replace('ó', 'o').replace('ú', 'u').replace('ê', 'e')
    
    mapa_grupos = {
        'graos e outras grandes culturas': 'Grãos',
        'graos': 'Grãos',
        'hortalicas': 'Hortaliças',
        'frutas': 'Frutas',
        'florestais': 'Florestais',
        'pecuaria': 'Pecuária',
        'flores': 'Flores',
        'a': 'Grãos',
        'b': 'Hortaliças',
        'c': 'Frutas',
        'd': 'Florestais',
        'e': 'Pecuária',
    }
    
    return mapa_grupos.get(grupo_str, 'Outros')

def classificar_cultura(cultura):
    """Classifica cultura em grupos"""
    if pd.isna(cultura):
        return 'Outros'
    
    cultura_upper = str(cultura).upper().strip()
    
    graos = ['SOJA', 'MILHO', 'TRIGO', 'FEIJAO', 'FEIJÃO', 'ARROZ', 'CANA']
    hortalicas = ['TOMATE', 'BATATA', 'CEBOLA', 'CENOURA', 'ALFACE', 'MANDIOCA']
    frutas = ['LARANJA', 'BANANA', 'UVA', 'MACA', 'MAÇÃ', 'MAMAO', 'MAMÃO']
    florestais = ['MADEIRA', 'PINUS', 'EUCALIPTO', 'ERVA-MATE']
    pecuaria = ['LEITE', 'BOVINO', 'SUINO', 'FRANGO', 'OVOS', 'PEIXE']
    flores = ['ROSA', 'CRISANTEMO', 'ORQUIDEA']
    
    for keyword in graos:
        if keyword in cultura_upper:
            return 'Grãos'
    for keyword in hortalicas:
        if keyword in cultura_upper:
            return 'Hortaliças'
    for keyword in frutas:
        if keyword in cultura_upper:
            return 'Frutas'
    for keyword in florestais:
        if keyword in cultura_upper:
            return 'Florestais'
    for keyword in pecuaria:
        if keyword in cultura_upper:
            return 'Pecuária'
    for keyword in flores:
        if keyword in cultura_upper:
            return 'Flores'
    
    return 'Outros'

def obter_keywords_coluna(coluna):
    """Retorna palavras-chave para identificar colunas"""
    keywords = {
        'MUNICIPIO': ['município', 'municipio', 'cidade'],
        'CULTURA': ['cultura', 'produto', 'cultivo'],
        'PRODUCAO': ['produção', 'producao', 'quant'],
        'AREA_PLANTADA': ['área', 'area', 'ha', 'hectare'],
        'VALOR_BRUTO': ['vbp', 'valor', 'bruto']
    }
    return keywords.get(coluna, [])

def padronizar_colunas(df, ano):
    """Padroniza nomes de colunas entre diferentes anos"""
    mapeamentos = {
        'município': 'MUNICIPIO',
        'municipio': 'MUNICIPIO',
        'produção': 'PRODUCAO',
        'producao': 'PRODUCAO',
        'área (ha)': 'AREA_PLANTADA',
        'area (ha)': 'AREA_PLANTADA',
        'área': 'AREA_PLANTADA',
        'area': 'AREA_PLANTADA',
        'vbp': 'VALOR_BRUTO',
        'valor bruto': 'VALOR_BRUTO',
        'cultura': 'CULTURA',
        'produto': 'CULTURA',
        'grupo': 'GRUPO_CULTURA',
        'grupo cultura': 'GRUPO_CULTURA',
        'região': 'REGIAO',
        'regiao': 'REGIAO'
    }
    
    df_renamed = df.copy()
    for col_original in df.columns:
        col_lower = str(col_original).lower().strip()
        if col_lower in mapeamentos:
            df_renamed = df_renamed.rename(columns={col_original: mapeamentos[col_lower]})
    
    colunas_essenciais = ['MUNICIPIO', 'CULTURA', 'PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO']
    for col in colunas_essenciais:
        if col not in df_renamed.columns:
            for df_col in df_renamed.columns:
                if any(keyword in str(df_col).lower() for keyword in obter_keywords_coluna(col)):
                    df_renamed = df_renamed.rename(columns={df_col: col})
                    break
    
    if 'GRUPO_CULTURA' not in df_renamed.columns and 'CULTURA' in df_renamed.columns:
        df_renamed['GRUPO_CULTURA'] = df_renamed['CULTURA'].apply(classificar_cultura)
    elif 'GRUPO_CULTURA' in df_renamed.columns:
        df_renamed['GRUPO_CULTURA'] = df_renamed['GRUPO_CULTURA'].apply(padronizar_grupo_existente)
        mask_vazio = df_renamed['GRUPO_CULTURA'].isna() | (df_renamed['GRUPO_CULTURA'] == 'Outros')
        if mask_vazio.any() and 'CULTURA' in df_renamed.columns:
            df_renamed.loc[mask_vazio, 'GRUPO_CULTURA'] = df_renamed.loc[mask_vazio, 'CULTURA'].apply(classificar_cultura)
    
    return df_renamed

def carregar_dados_multi_anos(pasta="data"):
    """Carrega dados de múltiplos anos de VBP"""
    print("\n📂 Buscando arquivos VBP...")
    
    patterns = ["VBP*.xls", "VBP*.xlsx", "vbp*.xls", "vbp*.xlsx"]
    arquivos_encontrados = []
    
    for pattern in patterns:
        arquivos_encontrados.extend(glob.glob(os.path.join(pasta, pattern)))
    
    # Remover duplicatas
    arquivos_encontrados = list(set(arquivos_encontrados))
    
    dados_por_ano = {}
    
    for arquivo in arquivos_encontrados:
        nome_arquivo = os.path.basename(arquivo).lower()
        match = re.search(r'(20\d{2})', nome_arquivo)
        if match:
            ano = int(match.group(1))
            
            # Evitar processar o mesmo ano duas vezes
            if ano in dados_por_ano:
                continue
                
            try:
                print(f"    ✓ Carregando: {os.path.basename(arquivo)} ({ano})")
                df = pd.read_excel(arquivo, skiprows=1)
                
                df.columns = [str(col).strip().replace('\n', '').replace('  ', ' ') 
                             for col in df.columns]
                
                df = padronizar_colunas(df, ano)
                df['ANO'] = ano
                
                dados_por_ano[ano] = df
                print(f"      {len(df):,} registros carregados")
                
            except Exception as e:
                print(f"    ✗ Erro ao carregar {ano}: {e}")
    
    return dados_por_ano

# =============================================================================
# FASE 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================

print("\n" + "=" * 100)
print("FASE 1: CARREGAMENTO E EXPLORAÇÃO DOS DADOS")
print("=" * 100)

dados_por_ano = carregar_dados_multi_anos()

if not dados_por_ano:
    print("\n❌ ERRO: Nenhum dado foi carregado. Verifique os arquivos.")
    exit()

print(f"\n✓ Total de anos carregados: {len(dados_por_ano)}")
print(f"✓ Período: {min(dados_por_ano.keys())} - {max(dados_por_ano.keys())}")

# Combinar todos os anos
print("\n🔄 Combinando dados de todos os anos...")
todos_dados = pd.concat(dados_por_ano.values(), ignore_index=True)

print(f"\n📊 Informações do dataset combinado:")
print(f"   • Total de registros: {len(todos_dados):,}")
if 'MUNICIPIO' in todos_dados.columns:
    print(f"   • Total de municípios: {todos_dados['MUNICIPIO'].nunique():,}")
print(f"   • Período: {todos_dados['ANO'].min()} - {todos_dados['ANO'].max()}")

# =============================================================================
# FASE 2: PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES
# =============================================================================

print("\n" + "=" * 100)
print("FASE 2: PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES")
print("=" * 100)

# Converter colunas numéricas
colunas_numericas = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO']
for col in colunas_numericas:
    if col in todos_dados.columns:
        todos_dados[col] = pd.to_numeric(todos_dados[col], errors='coerce')

print(f"   Registros antes da limpeza: {len(todos_dados):,}")
todos_dados = todos_dados.dropna(subset=colunas_numericas)
print(f"   Registros após limpeza: {len(todos_dados):,}")

# Engenharia de Features
print("\n🔧 Engenharia de Features:")
print("   • Calculando diversidade produtiva por município")
diversidade = todos_dados.groupby(['ANO', 'MUNICIPIO'])['CULTURA'].nunique().reset_index()
diversidade.columns = ['ANO', 'MUNICIPIO', 'diversidade_produtiva']
todos_dados = todos_dados.merge(diversidade, on=['ANO', 'MUNICIPIO'], how='left')

print("   • Criando features derivadas")
todos_dados['PRODUTIVIDADE'] = todos_dados['PRODUCAO'] / todos_dados['AREA_PLANTADA'].replace(0, 1)
todos_dados['VBP_POR_HA'] = todos_dados['VALOR_BRUTO'] / todos_dados['AREA_PLANTADA'].replace(0, 1)
todos_dados['INTENSIDADE_ECONOMICA'] = todos_dados['VALOR_BRUTO'] / todos_dados['PRODUCAO'].replace(0, 1)

# Criar variável target
print("\n🎯 Criando variável target: RISCO_DESPERDICIO")

def classificar_risco(df_ano):
    """Classifica risco de desperdício"""
    prod_q = df_ano['PRODUCAO'].quantile([0.33, 0.66])
    area_q = df_ano['AREA_PLANTADA'].quantile([0.33, 0.66])
    valor_q = df_ano['VALOR_BRUTO'].quantile([0.33, 0.66])
    div_q = df_ano['diversidade_produtiva'].quantile([0.33, 0.66])
    
    def calcular_score(row):
        score = 0
        if row['PRODUCAO'] <= prod_q.iloc[0]: score += 1
        elif row['PRODUCAO'] >= prod_q.iloc[1]: score -= 1
        
        if row['AREA_PLANTADA'] <= area_q.iloc[0]: score += 1
        elif row['AREA_PLANTADA'] >= area_q.iloc[1]: score -= 1
        
        if row['VALOR_BRUTO'] <= valor_q.iloc[0]: score += 1
        elif row['VALOR_BRUTO'] >= valor_q.iloc[1]: score -= 1
        
        if row['diversidade_produtiva'] <= div_q.iloc[0]: score += 1
        elif row['diversidade_produtiva'] >= div_q.iloc[1]: score -= 1
        
        if score >= 2: return 'ALTO'
        elif score <= -2: return 'BAIXO'
        else: return 'MEDIO'
    
    df_ano['RISCO_DESPERDICIO'] = df_ano.apply(calcular_score, axis=1)
    return df_ano

todos_dados = pd.concat([classificar_risco(df) for ano, df in todos_dados.groupby('ANO')], 
                        ignore_index=True)

print("\n📊 Distribuição de RISCO_DESPERDICIO:")
print(todos_dados['RISCO_DESPERDICIO'].value_counts())

# =============================================================================
# FASE 3: PREPARAÇÃO PARA MODELAGEM
# =============================================================================

print("\n" + "=" * 100)
print("FASE 3: PREPARAÇÃO PARA MODELAGEM")
print("=" * 100)

# Separação de features e target
features_numericas = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 
                     'diversidade_produtiva', 'PRODUTIVIDADE', 'VBP_POR_HA', 
                     'INTENSIDADE_ECONOMICA', 'ANO']
X = todos_dados[features_numericas + ['GRUPO_CULTURA']].copy()
y = todos_dados['RISCO_DESPERDICIO']

print(f"\n   • Features (X): {X.shape}")
print(f"   • Target (y): {y.shape}")

# One-Hot Encoding
print("\n🔄 Aplicando One-Hot Encoding...")
X = pd.get_dummies(X, drop_first=True, dtype='int64')
print(f"   • Features após encoding: {X.shape}")

# Divisão treino/teste
print("\n✂ Dividindo dados em treino (70%) e teste (30%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"   • Treino: {X_train.shape[0]:,} registros")
print(f"   • Teste: {X_test.shape[0]:,} registros")

# Normalização
print("\n🔄 Aplicando normalização MinMax...")
minmax = MinMaxScaler()

X_train_norm = pd.DataFrame(minmax.fit_transform(X_train), columns=X_train.columns)
X_test_norm = pd.DataFrame(minmax.transform(X_test), columns=X_test.columns)

print("   ✓ Dados normalizados!")

# =============================================================================
# FASE 4: SELEÇÃO DE FEATURES (APENAS MÉTODOS RÁPIDOS)
# =============================================================================

print("\n" + "=" * 100)
print("FASE 4: SELEÇÃO DE FEATURES (Métodos Rápidos)")
print("=" * 100)

# SelectKBest - RÁPIDO
print("\n🔍 SelectKBest (F-statistic) - OTIMIZADO")
skb = SelectKBest(f_classif, k=min(10, X_train.shape[1]))
skb.fit(X_train, y_train)
selected_skb = X_train.columns[skb.get_support()]
print(f"   Features selecionadas: {len(selected_skb)}")
print(f"   {list(selected_skb)}")
RESULTADOS['feature_selection']['SelectKBest'] = list(selected_skb)

# Feature Importance (Random Forest) - RÁPIDO
print("\n🔍 Feature Importance (Random Forest) - XAI")
rf_importance = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_importance.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_importance.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Features mais importantes:")
print(feature_importance.head(10).to_string(index=False))
RESULTADOS['feature_selection']['Importance'] = feature_importance.to_dict('records')[:15]

# Visualizar importância
plt.figure(figsize=(12, 6))
top_features = feature_importance.head(min(15, len(feature_importance)))
sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance - Random Forest (XAI)', fontsize=14, fontweight='bold')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
print("\n   ✓ Salvo: outputs/feature_importance.png")
plt.close()

# =============================================================================
# FASE 5: TREINAMENTO E COMPARAÇÃO DE MODELOS (OTIMIZADO)
# =============================================================================

print("\n" + "=" * 100)
print("FASE 5: TREINAMENTO E COMPARAÇÃO DE MODELOS")
print("=" * 100)

# Dicionário de modelos OTIMIZADOS
modelos = {
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
    'Naive Bayes': GaussianNB(),
}

# XGBoost se disponível
if XGBOOST_AVAILABLE:
    modelos['XGBoost'] = XGBClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1, eval_metric='mlogloss')

# Treinar e avaliar todos os modelos
resultados_comparacao = []

print("\n🤖 Treinando modelos...")
for nome, modelo in modelos.items():
    print(f"\n   Treinando {nome}...")
    start_time = time.time()
    
    try:
        # Treinar
        modelo.fit(X_train_norm, y_train)
        
        # Predizer
        y_pred = modelo.predict(X_test_norm)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation (3-fold para ser mais rápido)
        cv_scores = cross_val_score(modelo, X_train_norm, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        elapsed_time = time.time() - start_time
        
        # Armazenar resultados
        resultado = {
            'Modelo': nome,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'CV Mean': cv_mean,
            'CV Std': cv_std,
            'Tempo (s)': elapsed_time
        }
        resultados_comparacao.append(resultado)
        
        print(f"      ✓ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Tempo: {elapsed_time:.1f}s")
        
        RESULTADOS['models'][nome] = resultado
        
    except Exception as e:
        print(f"      ✗ Erro: {e}")

# Criar DataFrame de comparação
df_comparacao = pd.DataFrame(resultados_comparacao)
df_comparacao = df_comparacao.sort_values('Accuracy', ascending=False)

print("\n" + "=" * 100)
print("📊 COMPARAÇÃO DE TODOS OS MODELOS")
print("=" * 100)
print(df_comparacao.to_string(index=False))

# Salvar comparação
df_comparacao.to_csv('data/comparacao_modelos.csv', index=False)
print("\n✓ Salvo: data/comparacao_modelos.csv")

# =============================================================================
# VISUALIZAÇÃO DA COMPARAÇÃO
# =============================================================================

print("\n📊 Criando visualizações...")

# Gráfico 1: Comparação de Accuracy
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

df_sorted = df_comparacao.sort_values('Accuracy')
axes[0, 0].barh(df_sorted['Modelo'], df_sorted['Accuracy'], color='skyblue')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_title('Comparação de Accuracy', fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

df_sorted = df_comparacao.sort_values('F1-Score')
axes[0, 1].barh(df_sorted['Modelo'], df_sorted['F1-Score'], color='lightcoral')
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_title('Comparação de F1-Score', fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

df_sorted = df_comparacao.sort_values('Precision')
axes[1, 0].barh(df_sorted['Modelo'], df_sorted['Precision'], color='lightgreen')
axes[1, 0].set_xlabel('Precision')
axes[1, 0].set_title('Comparação de Precision', fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

df_sorted = df_comparacao.sort_values('CV Mean')
axes[1, 1].barh(df_sorted['Modelo'], df_sorted['CV Mean'], color='plum')
axes[1, 1].set_xlabel('CV Score')
axes[1, 1].set_title('Cross-Validation', fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/comparacao_metricas.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: outputs/comparacao_metricas.png")
plt.close()

# Matriz de confusão do melhor modelo
melhor_modelo_nome = df_comparacao.iloc[0]['Modelo']
melhor_modelo = modelos[melhor_modelo_nome]
y_pred_melhor = melhor_modelo.predict(X_test_norm)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_melhor)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['ALTO', 'BAIXO', 'MEDIO'],
            yticklabels=['ALTO', 'BAIXO', 'MEDIO'])
plt.title(f'Matriz de Confusão - {melhor_modelo_nome}', fontsize=14, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix_melhor.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: outputs/confusion_matrix_melhor.png")
plt.close()

# =============================================================================
# ANÁLISE TEMPORAL
# =============================================================================

print("\n" + "=" * 100)
print("FASE 6: ANÁLISE TEMPORAL")
print("=" * 100)

# Agregar dados por ano
dados_anuais = todos_dados.groupby('ANO').agg({
    'MUNICIPIO': 'nunique',
    'PRODUCAO': 'sum',
    'AREA_PLANTADA': 'sum',
    'VALOR_BRUTO': 'sum',
    'diversidade_produtiva': 'mean',
    'PRODUTIVIDADE': 'mean',
    'VBP_POR_HA': 'mean'
}).reset_index()

# Evolução do risco por ano
risco_por_ano = todos_dados.groupby(['ANO', 'RISCO_DESPERDICIO']).size().unstack(fill_value=0)
risco_pct_ano = risco_por_ano.div(risco_por_ano.sum(axis=1), axis=0) * 100

# Visualização temporal
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].plot(dados_anuais['ANO'], dados_anuais['VALOR_BRUTO'] / 1e9, marker='o', linewidth=2)
axes[0, 0].set_title('Evolução do VBP Total', fontweight='bold')
axes[0, 0].set_ylabel('VBP (Bilhões R$)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(dados_anuais['ANO'], dados_anuais['PRODUCAO'] / 1e6, marker='o', linewidth=2, color='green')
axes[0, 1].set_title('Evolução da Produção Total', fontweight='bold')
axes[0, 1].set_ylabel('Produção (Milhões)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(dados_anuais['ANO'], dados_anuais['diversidade_produtiva'], marker='o', linewidth=2, color='purple')
axes[1, 0].set_title('Diversidade Produtiva', fontweight='bold')
axes[1, 0].set_ylabel('Diversidade Média')
axes[1, 0].set_xlabel('Ano')
axes[1, 0].grid(True, alpha=0.3)

risco_pct_ano.plot(kind='bar', stacked=True, ax=axes[1, 1], color=['#FF6B6B', '#4ECDC4', '#FFE66D'])
axes[1, 1].set_title('Distribuição de Risco por Ano', fontweight='bold')
axes[1, 1].set_ylabel('Percentual (%)')
axes[1, 1].set_xlabel('Ano')
axes[1, 1].legend(title='Risco')

plt.tight_layout()
plt.savefig('outputs/evolucao_temporal.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: outputs/evolucao_temporal.png")
plt.close()

# =============================================================================
# SALVAR RESULTADOS
# =============================================================================

print("\n💾 Salvando resultados...")
with open('data/resultados_ml.json', 'w', encoding='utf-8') as f:
    json.dump(RESULTADOS, f, indent=4, ensure_ascii=False, default=str)
print("   ✓ Salvo: data/resultados_ml.json")

# =============================================================================
# RESUMO FINAL
# =============================================================================

print("\n" + "=" * 100)
print("✅ ANÁLISE COMPLETA FINALIZADA")
print("=" * 100)

print(f"\n📊 RESUMO:")
print(f"   • {len(modelos)} modelos treinados e comparados")
print(f"   • Melhor modelo: {melhor_modelo_nome} (Accuracy: {df_comparacao.iloc[0]['Accuracy']:.4f})")
print(f"   • {len(dados_por_ano)} anos de dados analisados ({min(dados_por_ano.keys())}-{max(dados_por_ano.keys())})")
print(f"   • {len(todos_dados):,} registros processados")

print("\n📁 ARQUIVOS GERADOS:")
arquivos = [
    'outputs/feature_importance.png',
    'outputs/comparacao_metricas.png',
    'outputs/confusion_matrix_melhor.png',
    'outputs/evolucao_temporal.png',
    'data/comparacao_modelos.csv',
    'data/resultados_ml.json'
]
for arquivo in arquivos:
    print(f"   ✓ {arquivo}")

print("\n🎯 PRÓXIMO PASSO:")
print("   Abra o arquivo: views/dashboard_final.html")

print("\n" + "=" * 100)
