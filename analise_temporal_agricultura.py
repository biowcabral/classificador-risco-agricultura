"""
Sistema de Análise Temporal de Risco de Desperdício Agrícola
Aplicação de Machine Learning para análise de dados agrícolas
Autor: Classificador de Risco Agrícola
Data: Novembro 2025
"""

# =============================================================================
# FASE 1: IMPORTAÇÃO E EXPLORAÇÃO DOS DADOS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                            ConfusionMatrixDisplay)

import glob
import os
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("  SISTEMA DE ANALISE TEMPORAL DE RISCO AGRICOLA")
print("  Análise Preditiva de Desperdício na Agricultura")
print("=" * 80)

# =============================================================================
# FASE 1.1: CARREGAMENTO DE DADOS MULTI-ANUAIS
# =============================================================================

print("\nFASE 1: EXPLORACAO DOS DADOS")
print("-" * 80)

def carregar_dados_multi_anos(pasta="."):
    """
    Carrega dados de múltiplos anos de VBP (Valor Bruto da Produção)
    Retorna dicionário com dados organizados por ano
    """
    print("\nBuscando arquivos VBP...")
    
    # Buscar todos os arquivos VBP
    patterns = ["VBP*.xls", "VBP*.xlsx", "vbp*.xls", "vbp*.xlsx"]
    arquivos_encontrados = []
    
    for pattern in patterns:
        arquivos_encontrados.extend(glob.glob(os.path.join(pasta, pattern)))
    
    # Organizar por ano
    dados_por_ano = {}
    
    for arquivo in arquivos_encontrados:
        nome_arquivo = os.path.basename(arquivo).lower()
        # Extrair ano do nome do arquivo
        match = re.search(r'(20\d{2})', nome_arquivo)
        if match:
            ano = int(match.group(1))
            try:
                print(f"    Carregando: {os.path.basename(arquivo)} ({ano})")
                df = pd.read_excel(arquivo, skiprows=1)
                
                # Limpar nomes das colunas
                df.columns = [col.strip().replace('\n', '').replace('  ', ' ') 
                             for col in df.columns]
                
                # Padronizar colunas
                df = padronizar_colunas(df, ano)
                df['ANO'] = ano
                
                dados_por_ano[ano] = df
                print(f"   {len(df)} registros carregados para {ano}")
                
            except Exception as e:
                print(f"    Erro ao carregar {ano}: {e}")
    
    return dados_por_ano

def padronizar_colunas(df, ano):
    """
    Padroniza nomes de colunas entre diferentes anos
    Garante consistência nos dados históricos
    """
    # Mapeamento de colunas
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
    
    # Aplicar mapeamentos
    df_renamed = df.copy()
    for col_original in df.columns:
        col_lower = col_original.lower().strip()
        if col_lower in mapeamentos:
            df_renamed = df_renamed.rename(columns={col_original: mapeamentos[col_lower]})
    
    # Garantir colunas essenciais
    colunas_essenciais = ['MUNICIPIO', 'CULTURA', 'PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO']
    for col in colunas_essenciais:
        if col not in df_renamed.columns:
            # Tentar inferir
            for df_col in df_renamed.columns:
                if any(keyword in df_col.lower() for keyword in obter_keywords_coluna(col)):
                    df_renamed = df_renamed.rename(columns={df_col: col})
                    break
    
    # Criar GRUPO_CULTURA se não existir
    if 'GRUPO_CULTURA' not in df_renamed.columns and 'CULTURA' in df_renamed.columns:
        df_renamed['GRUPO_CULTURA'] = df_renamed['CULTURA'].apply(classificar_cultura)
    elif 'GRUPO_CULTURA' in df_renamed.columns:
        # Padronizar grupos existentes (2023-2024 têm nomes diferentes)
        df_renamed['GRUPO_CULTURA'] = df_renamed['GRUPO_CULTURA'].apply(padronizar_grupo_existente)
        # Se ainda houver valores vazios, classificar pela cultura
        mask_vazio = df_renamed['GRUPO_CULTURA'].isna() | (df_renamed['GRUPO_CULTURA'] == 'Outros')
        if mask_vazio.any() and 'CULTURA' in df_renamed.columns:
            df_renamed.loc[mask_vazio, 'GRUPO_CULTURA'] = df_renamed.loc[mask_vazio, 'CULTURA'].apply(classificar_cultura)
    
    return df_renamed

def padronizar_grupo_existente(grupo):
    """
    Padroniza nomes de grupos que já existem nos dados
    Converte Sistema 1 (A-E) e Sistema 2 (nomes) para formato unificado
    """
    if pd.isna(grupo):
        return 'Outros'
    
    grupo_str = str(grupo).strip().lower()
    
    # Remover acentos comuns para matching
    grupo_str = grupo_str.replace('ã', 'a').replace('õ', 'o').replace('ç', 'c')
    grupo_str = grupo_str.replace('á', 'a').replace('é', 'e').replace('í', 'i')
    grupo_str = grupo_str.replace('ó', 'o').replace('ú', 'u').replace('ê', 'e')
    
    # Mapeamento direto
    mapa_grupos = {
        'graos e outras grandes culturas': 'Grãos',
        'graos': 'Grãos',
        'hortalicas': 'Hortaliças',
        'frutas': 'Frutas',
        'florestais': 'Florestais',
        'pecuaria': 'Pecuária',
        'flores': 'Flores',
        'a': 'Grãos',       # Sistema antigo
        'b': 'Hortaliças',  # Sistema antigo
        'c': 'Frutas',      # Sistema antigo
        'd': 'Florestais',  # Sistema antigo
        'e': 'Pecuária',    # Sistema antigo
    }
    
    return mapa_grupos.get(grupo_str, 'Outros')

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

def classificar_cultura(cultura):
    """
    Classifica cultura em grupos unificados
    Sistema aprimorado que cobre todas as principais culturas
    """
    if pd.isna(cultura):
        return 'Outros'
    
    cultura_upper = str(cultura).upper().strip()
    
    # GRÃOS E GRANDES CULTURAS
    graos_keywords = ['SOJA', 'MILHO', 'TRIGO', 'FEIJAO', 'FEIJÃO', 'AVEIA', 'CEVADA', 
                      'TRITICALE', 'ARROZ', 'SORGO', 'GIRASSOL', 'CANOLA', 'AMENDOIM',
                      'PAINCO', 'TRIGUILHO', 'AZEVEM', 'CENTEIO', 'CANA', 'ALGODAO',
                      'ALGODÃO', 'MAMONA', 'LINHO']
    
    # HORTALIÇAS
    hortalicas_keywords = ['TOMATE', 'BATATA', 'CEBOLA', 'CENOURA', 'ALFACE', 'REPOLHO',
                           'BETERRABA', 'COUVE', 'PEPINO', 'PIMENTAO', 'PIMENTÃO', 'ABOBORA',
                           'ABÓBORA', 'ABOBRINHA', 'QUIABO', 'BERINJELA', 'PIMENTA', 'ALHO',
                           'BROCOLIS', 'BRÓCOLIS', 'COUVE-FLOR', 'MANDIOCA', 'AIPIM',
                           'MANDIOQUINHA', 'BATATA-DOCE', 'CHUCHU', 'ERVILHA', 'VAGEM',
                           'AGRIAO', 'AGRIÃO', 'SALSA', 'CEBOLINHA', 'RUCULA', 'RÚCULA',
                           'ESPINAFRE', 'CHICORIA', 'CHICÓRIA', 'ACELGA', 'RABANETE', 'NABO',
                           'MORANGA', 'MELANCIA', 'MELAO', 'MELÃO', 'MAXIXE', 'JILO', 'JILÓ',
                           'INHAME', 'CARÁ', 'GENGIBRE', 'ALCACHOFRA', 'ASPARGO', 'PALMITO',
                           'COGUMELO', 'SALSAO', 'SALSÃO']
    
    # FRUTAS
    frutas_keywords = ['LARANJA', 'BANANA', 'UVA', 'MACA', 'MAÇÃ', 'MAMAO', 'MAMÃO', 'MANGA',
                       'ABACAXI', 'LIMAO', 'LIMÃO', 'TANGERINA', 'BERGAMOTA', 'MEXERICA',
                       'PESSEGO', 'PÊSSEGO', 'AMEIXA', 'MORANGO', 'MARACUJA', 'MARACUJÁ',
                       'GOIABA', 'ABACATE', 'CAQUI', 'FIGO', 'PERA', 'KIWI', 'ACEROLA',
                       'JABUTICABA', 'PITANGA', 'CAJU', 'NECTARINA', 'LICHIA', 'CARAMBOLA',
                       'FRAMBOESA', 'AMORA', 'MIRTILO', 'CEREJA', 'ROMA', 'ROMÃ', 'JACA',
                       'PITAYA', 'NOZ', 'CASTANHA', 'PECAN', 'MACADAMIA', 'CAFE', 'CAFÉ']
    
    # FLORESTAIS
    florestais_keywords = ['MADEIRA', 'PINUS', 'EUCALIPTO', 'PINHEIRO', 'LENHA', 'BRACATINGA',
                           'ERVA-MATE', 'PINHAO', 'PINHÃO', 'SERINGUEIRA', 'LATEX', 'LÁTEX',
                           'RESINA', 'MATA', 'FLORESTAL', 'RESIDUOS FLORESTAIS']
    
    # PECUÁRIA
    pecuaria_keywords = ['LEITE', 'BOVINO', 'BOI', 'VACA', 'BEZERRO', 'GARROTE', 'NOVILHO',
                         'NOVILHA', 'TOURO', 'VITELO', 'SUINO', 'SUÍNO', 'PORCO', 'LEITAO',
                         'LEITÃO', 'FRANGO', 'GALINHA', 'PINTINHO', 'AVES', 'AVE', 'OVOS',
                         'OVO', 'PERU', 'PATO', 'MARRECO', 'CODORNA', 'OVINO', 'CAPRINO',
                         'CABRA', 'CARNEIRO', 'CORDEIRO', 'EQUINO', 'CAVALO', 'EGUA', 'ÉGUA',
                         'POTRO', 'MUAR', 'BURRO', 'MULA', 'PEIXE', 'TILAPIA', 'TILÁPIA',
                         'PACU', 'TAMBAQUI', 'TAMBACU', 'CARPA', 'TRAIRA', 'TRAÍRA', 'PIAUCU',
                         'PESCADO', 'CAMARAO', 'CAMARÃO', 'OSTRA', 'MEXILHAO', 'MEXILHÃO',
                         'MEL', 'PROPOLIS', 'PRÓPOLIS', 'GELEIA', 'POLEM', 'PÓLEN',
                         'SERICICULTURA', 'BICHO-DA-SEDA', 'CASULO', 'SILAGEM', 'FENO',
                         'PASTAGEM', 'FORRAGEM', 'LÃ', 'LA']
    
    # FLORES
    flores_keywords = ['ROSA', 'CRISANTEMO', 'CRAVO', 'GERBERA', 'ORQUIDEA', 'ORQUÍDEA',
                       'ANTHURIO', 'ANTÚRIO', 'ASTROMELIA', 'ASTROMÉLIA', 'LIRIO', 'LÍRIO',
                       'VIOLETA', 'AZALEIA', 'AZALÉIA', 'BEGONIA', 'BEGÔNIA', 'BRINCO',
                       'PETUNIA', 'PETÚNIA', 'AMOR-PERFEITO', 'MARGARIDA', 'TAGETE',
                       'SAMAMBAIA', 'SUCULENTA', 'KALANCHOE', 'PRIMAVERA', 'CAMELIA',
                       'CAMÉLIA', 'MOSQUITINHO', 'SOLIDASTER', 'LAVANDA', 'SALVIA', 'SÁLVIA',
                       'PLANTAS ORNAMENTAIS', 'GRAMADO', 'GRAMA']
    
    # Verificar em ordem de prioridade
    for keyword in graos_keywords:
        if keyword in cultura_upper:
            return 'Grãos'
    
    for keyword in hortalicas_keywords:
        if keyword in cultura_upper:
            return 'Hortaliças'
    
    for keyword in frutas_keywords:
        if keyword in cultura_upper:
            return 'Frutas'
    
    for keyword in florestais_keywords:
        if keyword in cultura_upper:
            return 'Florestais'
    
    for keyword in pecuaria_keywords:
        if keyword in cultura_upper:
            return 'Pecuária'
    
    for keyword in flores_keywords:
        if keyword in cultura_upper:
            return 'Flores'
    
    return 'Outros'

# Carregar dados
dados_por_ano = carregar_dados_multi_anos()

if not dados_por_ano:
    print("\n ERRO: Nenhum dado foi carregado. Verifique os arquivos.")
    exit()

print(f"\n Total de anos carregados: {len(dados_por_ano)}")
print(f" Período: {min(dados_por_ano.keys())} - {max(dados_por_ano.keys())}")

# =============================================================================
# FASE 1.2: EXPLORAÇÃO INICIAL DOS DADOS
# =============================================================================

print("\n" + "=" * 80)
print(" ANÁLISE EXPLORATÓRIA")
print("=" * 80)

# Combinar todos os anos em um único dataframe
print("\n Combinando dados de todos os anos...")
todos_dados = pd.concat(dados_por_ano.values(), ignore_index=True)

print(f"\n Informações do dataset combinado:")
print(f"   • Total de registros: {len(todos_dados):,}")
print(f"   • Colunas disponíveis: {list(todos_dados.columns)}")
if 'MUNICIPIO' in todos_dados.columns:
    print(f"   • Total de municípios: {todos_dados['MUNICIPIO'].nunique():,}")
print(f"   • Período: {todos_dados['ANO'].min()} - {todos_dados['ANO'].max()}")

# Visualização inicial dos dados
print("\n Primeiras linhas do dataset:")
print(todos_dados.head())

# Informações estruturais do dataset
print("\n Informações do dataset (df.info()):")
print(todos_dados.info())

# Análise estatística descritiva
print("\n Estatísticas descritivas (df.describe()):")
print(todos_dados.describe())

# Verificar valores ausentes
print("\n Valores ausentes:")
print(todos_dados.isnull().sum())

# =============================================================================
# FASE 2: PRÉ-PROCESSAMENTO DOS DADOS - TRANSFORMAÇÃO
# =============================================================================

print("\n" + "=" * 80)
print("FASE 2: PRE-PROCESSAMENTO DOS DADOS")
print("=" * 80)

# Converter colunas numéricas
print("\nConvertendo colunas numericas...")
colunas_numericas = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO']

for col in colunas_numericas:
    if col in todos_dados.columns:
        todos_dados[col] = pd.to_numeric(todos_dados[col], errors='coerce')

# Remover valores nulos
print(f"   Registros antes da limpeza: {len(todos_dados):,}")
todos_dados = todos_dados.dropna(subset=colunas_numericas)
print(f"   Registros após limpeza: {len(todos_dados):,}")

# Calcular feature de diversidade (engenharia de features)
print("\nEngenharia de Features...")
print("   • Calculando diversidade produtiva por município")
diversidade = todos_dados.groupby(['ANO', 'MUNICIPIO'])['CULTURA'].nunique().reset_index()
diversidade.columns = ['ANO', 'MUNICIPIO', 'diversidade_produtiva']
todos_dados = todos_dados.merge(diversidade, on=['ANO', 'MUNICIPIO'], how='left')

# Criar features derivadas
print("   • Criando features derivadas")
todos_dados['PRODUTIVIDADE'] = todos_dados['PRODUCAO'] / todos_dados['AREA_PLANTADA'].replace(0, 1)
todos_dados['VBP_POR_HA'] = todos_dados['VALOR_BRUTO'] / todos_dados['AREA_PLANTADA'].replace(0, 1)

# Criar variável target: RISCO_DESPERDICIO
print("\n Criando variável target: RISCO_DESPERDICIO")
print("   (baseado em quantis de produção, área, VBP e diversidade)")

def classificar_risco(df_ano):
    """
    Classifica risco de desperdício
    # Classificação baseada em quantis múltiplos
    """
    # Calcular quantis por ano
    prod_q = df_ano['PRODUCAO'].quantile([0.33, 0.66])
    area_q = df_ano['AREA_PLANTADA'].quantile([0.33, 0.66])
    valor_q = df_ano['VALOR_BRUTO'].quantile([0.33, 0.66])
    div_q = df_ano['diversidade_produtiva'].quantile([0.33, 0.66])
    
    def calcular_score(row):
        score = 0
        # Produção baixa = risco maior
        if row['PRODUCAO'] <= prod_q.iloc[0]: score += 1
        elif row['PRODUCAO'] >= prod_q.iloc[1]: score -= 1
        
        # Área baixa = risco maior
        if row['AREA_PLANTADA'] <= area_q.iloc[0]: score += 1
        elif row['AREA_PLANTADA'] >= area_q.iloc[1]: score -= 1
        
        # VBP baixo = risco maior
        if row['VALOR_BRUTO'] <= valor_q.iloc[0]: score += 1
        elif row['VALOR_BRUTO'] >= valor_q.iloc[1]: score -= 1
        
        # Diversidade baixa = risco maior
        if row['diversidade_produtiva'] <= div_q.iloc[0]: score += 1
        elif row['diversidade_produtiva'] >= div_q.iloc[1]: score -= 1
        
        # Classificar
        if score >= 2: return 'ALTO'
        elif score <= -2: return 'BAIXO'
        else: return 'MEDIO'
    
    df_ano['RISCO_DESPERDICIO'] = df_ano.apply(calcular_score, axis=1)
    return df_ano

# Aplicar classificação de risco para cada ano
print("   • Aplicando classificação de risco...")
todos_dados = pd.concat([classificar_risco(df) for ano, df in todos_dados.groupby('ANO')], 
                        ignore_index=True)

# Verificar distribuição do risco
print("\n Distribuição de RISCO_DESPERDICIO:")
print(todos_dados['RISCO_DESPERDICIO'].value_counts())
print("\nPercentuais:")
print(todos_dados['RISCO_DESPERDICIO'].value_counts(normalize=True) * 100)

# =============================================================================
# FASE 2.1: PREPARAÇÃO DOS DADOS PARA MODELAGEM
# =============================================================================

print("\n" + "=" * 80)
print(" PREPARAÇÃO DOS DADOS PARA MODELAGEM")
print("=" * 80)

# Separação entre variáveis preditoras (X) e variável alvo (y)
print("\nSeparando features (X) e target (y)...")
features_numericas = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 
                     'diversidade_produtiva', 'PRODUTIVIDADE', 'VBP_POR_HA', 'ANO']
X = todos_dados[features_numericas + ['GRUPO_CULTURA']].copy()
y = todos_dados['RISCO_DESPERDICIO']

print(f"   • Features (X): {X.shape}")
print(f"   • Target (y): {y.shape}")
print(f"\n   Colunas de X:")
print(f"   {list(X.columns)}")

# Aplicar One-Hot Encoding nas variáveis categóricas
# Transformação de variáveis categóricas em binárias (one-hot encoding)
print("\nAplicando binarizacao (One-Hot Encoding)...")
print("   Usando pd.get_dummies() nas variáveis categóricas")
X = pd.get_dummies(X, drop_first=True, dtype='int64')
print(f"   • Features após binarização: {X.shape}")
print(f"   • Novas colunas: {list(X.columns)}")

# Divisão estratificada em conjuntos de treino e teste
print("\nDividindo dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"   • Treino: {X_train.shape[0]:,} registros")
print(f"   • Teste: {X_test.shape[0]:,} registros")
print(f"\n   Distribuição do target no conjunto de treino:")
print(y_train.value_counts())

# Normalização dos dados usando MinMaxScaler (escala 0-1)
print("\nAplicando normalizacao (MinMaxScaler)...")
print("   IMPORTANTE: fit() apenas no treino, transform() em ambos!")

minmax = MinMaxScaler()

# Fit apenas no treino!
minmax.fit(X_train)

# Transform em ambos
X_train_norm = minmax.transform(X_train)
X_test_norm = minmax.transform(X_test)

# Converter de volta para DataFrame (facilita visualização)
X_train_norm = pd.DataFrame(X_train_norm, columns=X_train.columns)
X_test_norm = pd.DataFrame(X_test_norm, columns=X_test.columns)

print(f"   Dados normalizados!")
print(f"   - X_train_norm: {X_train_norm.shape}")
print(f"   - X_test_norm: {X_test_norm.shape}")

print("\nAmostra dos dados normalizados:")
print(X_train_norm.head())

# =============================================================================
# FASE 3: TREINAMENTO DO MODELO DE MACHINE LEARNING
# =============================================================================

print("\n" + "=" * 80)
print("FASE 3: TREINAMENTO DO MODELO")
print("=" * 80)

# -------------------------------------------------------------------------
# Random Forest (Floresta Aleatoria)
# -------------------------------------------------------------------------
print("\n" + "-" * 80)
print("Modelo: Random Forest (Floresta Aleatoria)")
print("-" * 80)

print("\nTreinando modelo Random Forest com 100 arvores...")

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print(f"\nAcuracia:")
print(f"   {acuracia:.4f} ({acuracia*100:.2f}%)")

print("\nRelatorio de Classificacao:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusao:")
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=['ALTO', 'BAIXO', 'MEDIO']).plot()
plt.title('Random Forest - Matriz de Confusao')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png', dpi=150, bbox_inches='tight')
print("   Salvo: confusion_matrix_rf.png")
plt.close()

print("\nImportancia das Features (Top 10):")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10))

print("\n" + "=" * 80)
print("FASE 3 CONCLUIDA")
print("=" * 80)
print(f"\nModelo treinado: Random Forest")
print(f"Acuracia alcancada: {acuracia:.4f} ({acuracia*100:.2f}%)")

# =============================================================================
# FASE 4: ANALISE TEMPORAL E EVOLUCAO DE RISCO
# =============================================================================

print("\n" + "=" * 80)
print(" FASE 4: ANALISE TEMPORAL E EVOLUCAO DE RISCO")
print("=" * 80)

print("\nAnalisando tendencias ao longo dos anos...")

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

dados_anuais.columns = ['ANO', 'MUNICIPIOS', 'PRODUCAO_TOTAL', 'AREA_TOTAL',
                        'VBP_TOTAL', 'DIVERSIDADE_MEDIA', 'PRODUTIVIDADE_MEDIA', 'VBP_HA_MEDIO']

print("\nEvolucao Anual dos Indicadores:")
print(dados_anuais)

# Evolução do risco por ano
risco_por_ano = todos_dados.groupby(['ANO', 'RISCO_DESPERDICIO']).size().unstack(fill_value=0)
risco_pct_ano = risco_por_ano.div(risco_por_ano.sum(axis=1), axis=0) * 100

print("\nDistribuicao de Risco por Ano (%):")
print(risco_pct_ano)

# Visualizar evolução temporal
print("\nCriando visualizacoes temporais...")

# Gráfico 1: Evolução do VBP
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(dados_anuais['ANO'], dados_anuais['VBP_TOTAL'] / 1e9, marker='o', linewidth=2)
axes[0, 0].set_title('Evolução do VBP Total', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Ano')
axes[0, 0].set_ylabel('VBP (Bilhões R$)')
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Evolução da Produção
axes[0, 1].plot(dados_anuais['ANO'], dados_anuais['PRODUCAO_TOTAL'] / 1e6, marker='o', linewidth=2, color='green')
axes[0, 1].set_title('Evolução da Produção Total', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Ano')
axes[0, 1].set_ylabel('Produção (Milhões ton)')
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Evolução da Diversidade
axes[1, 0].plot(dados_anuais['ANO'], dados_anuais['DIVERSIDADE_MEDIA'], marker='o', linewidth=2, color='purple')
axes[1, 0].set_title('Evolução da Diversidade Produtiva', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Ano')
axes[1, 0].set_ylabel('Diversidade Média')
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Evolução do Risco
risco_pct_ano.plot(kind='bar', stacked=True, ax=axes[1, 1], 
                   color=['#FF6B6B', '#FFE66D', '#4ECDC4'])
axes[1, 1].set_title('Evolução da Distribuição de Risco', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Ano')
axes[1, 1].set_ylabel('Percentual (%)')
axes[1, 1].legend(title='Risco', bbox_to_anchor=(1.05, 1))
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('evolucao_temporal.png', dpi=150, bbox_inches='tight')
print("   Salvo: evolucao_temporal.png")
plt.close()

# Análise por grupo de cultura ao longo do tempo
print("\nAnalisando evolucao por grupo de cultura...")
cultura_ano = todos_dados.groupby(['ANO', 'GRUPO_CULTURA'])['VALOR_BRUTO'].sum().unstack(fill_value=0)

plt.figure(figsize=(12, 6))
for cultura in cultura_ano.columns:
    plt.plot(cultura_ano.index, cultura_ano[cultura] / 1e9, marker='o', label=cultura, linewidth=2)
plt.title('Evolução do VBP por Grupo de Cultura', fontsize=16, fontweight='bold')
plt.xlabel('Ano', fontsize=12)
plt.ylabel('VBP (Bilhões R$)', fontsize=12)
plt.legend(title='Grupo de Cultura', bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evolucao_por_cultura.png', dpi=150, bbox_inches='tight')
print("   Salvo: evolucao_por_cultura.png")
plt.close()

# Correlação entre variáveis
print("\nAnalisando correlacoes...")
plt.figure(figsize=(10, 8))
correlacao = dados_anuais[['PRODUCAO_TOTAL', 'AREA_TOTAL', 'VBP_TOTAL', 
                           'DIVERSIDADE_MEDIA', 'PRODUTIVIDADE_MEDIA']].corr()
sns.heatmap(correlacao, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Indicadores Anuais', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlacao_indicadores.png', dpi=150, bbox_inches='tight')
print("   Salvo: correlacao_indicadores.png")
plt.close()

print("\n" + "=" * 80)
print("FASE 4 CONCLUIDA")
print("=" * 80)

# =============================================================================
# RESUMO FINAL
# =============================================================================

# Calculo de estatisticas finais
periodo_anos = dados_anuais['ANO'].max() - dados_anuais['ANO'].min() + 1
primeiro_ano = int(dados_anuais['ANO'].min())
ultimo_ano = int(dados_anuais['ANO'].max())

print("\n" + "=" * 80)
print("ANALISE COMPLETA FINALIZADA")
print("=" * 80)

print("\nRESUMO DA EXECUCAO:")
print(f"   Dados de {periodo_anos} anos processados ({primeiro_ano}-{ultimo_ano})")
print(f"   {len(todos_dados):,} registros analisados")
if 'MUNICIPIO' in todos_dados.columns:
    print(f"   {todos_dados['MUNICIPIO'].nunique():,} municipios unicos")
print(f"   1 modelo de ML treinado: Random Forest")
print(f"   Acuracia alcancada: {acuracia:.2%}")

print("\nARQUIVOS GERADOS:")
print("   - confusion_matrix_rf.png")
print("   - evolucao_temporal.png")
print("   - evolucao_por_cultura.png")
print("   - correlacao_indicadores.png")

print("\nCOMO VISUALIZAR:")
print("   Todas as visualizacoes foram salvas como imagens PNG")
print("   As imagens podem ser incluidas em apresentacoes e relatorios")

print("\nPROXIMOS PASSOS SUGERIDOS:")
print("   - Analisar os graficos de evolucao temporal gerados")
print("   - Examinar insights por grupo de cultura")
print("   - Investigar municipios de alto risco especificos")
print("   - Implementar sistema de monitoramento continuo")

print("\n" + "=" * 80)
print("Metodologia: Analise Exploratoria -> Pre-processamento -> Modelagem -> Validacao -> Visualizacao")
print("Modelo: Random Forest com validacao holdout (70/30)")
print("=" * 80)

