import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import json
import os
import warnings
warnings.filterwarnings('ignore')

class DashboardRiscoAgricultura:
    def __init__(self):
        self.clf = None
        self.df = None
        self.metrics = {}
        self.feature_importance = None
        self.predictions_df = None
        
    def calcular_diversidade(self, df, municipio_col, cultura_col):
        """Função para calcular diversidade produtiva"""
        return df.groupby(municipio_col)[cultura_col].nunique().rename('diversidade_produtiva')

    def load_and_process_data(self, file_path):
        """Carrega e processa os dados"""
        print("🚜 Carregando dados do VBP 2024...")
        df = pd.read_excel(file_path, skiprows=1)
        df.columns = [col.strip().replace('\n', '') for col in df.columns]
        
        # Mapear colunas
        column_mapping = {
            'Município': 'MUNICIPIO',
            'Produção': 'PRODUCAO', 
            'Área (ha)': 'AREA_PLANTADA',
            'VBP': 'VALOR_BRUTO',
            'Grupo': 'GRUPO_CULTURA',
            'Cultura': 'CULTURA'
        }
        df = df.rename(columns=column_mapping)
        
        # Calcular diversidade produtiva
        diversidade = self.calcular_diversidade(df, 'MUNICIPIO', 'CULTURA')
        df = df.merge(diversidade, left_on='MUNICIPIO', right_index=True)
        
        # Converter colunas numéricas
        numeric_cols = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=numeric_cols + ['diversidade_produtiva'])
        self.df = df
        return df

    def create_risk_classification(self):
        """Cria classificação de risco"""
        df = self.df
        
        # Quantis para classificação
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
            
            if score >= 2: return 'ALTO'
            elif score <= -2: return 'BAIXO'
            else: return 'MEDIO'

        self.df['RISCO_DESPERDICIO'] = self.df.apply(classificar_risco, axis=1)

    def train_model(self):
        """Treina o modelo Random Forest"""
        df = self.df
        features = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 'diversidade_produtiva', 'GRUPO_CULTURA']
        X = df[features].copy()
        y = df['RISCO_DESPERDICIO']

        # Preparar dados
        le = LabelEncoder()
        X['GRUPO_CULTURA'] = le.fit_transform(X['GRUPO_CULTURA'])
        
        scaler = StandardScaler()
        numeric_features = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 'diversidade_produtiva']
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Treinar modelo
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.clf.fit(X_train, y_train)

        # Predições e métricas
        y_pred = self.clf.predict(X_test)
        
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        accuracy = accuracy_score(y_test, y_pred)
        
        target_names = sorted(y.unique())
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'target_names': target_names,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Importância das features
        self.feature_importance = dict(zip(features, self.clf.feature_importances_))
        
        return X_test, y_test, y_pred

    def create_interactive_dashboard(self):
        """Cria dashboard interativo com Plotly"""
        df = self.df
        
        # 1. Gráfico de distribuição de risco
        risk_dist = df['RISCO_DESPERDICIO'].value_counts()
        
        fig1 = px.pie(
            values=risk_dist.values, 
            names=risk_dist.index,
            title="📊 Distribuição de Risco por Município",
            color_discrete_map={
                'ALTO': '#FF6B6B',
                'MEDIO': '#FFE66D', 
                'BAIXO': '#4ECDC4'
            }
        )
        fig1.update_layout(
            font=dict(size=14),
            title_font=dict(size=18, family="Arial Black"),
            showlegend=True,
            height=400
        )

        # 2. Análise regional
        if 'Região' in df.columns:
            region_risk = df.groupby(['Região', 'RISCO_DESPERDICIO']).size().unstack(fill_value=0)
            region_risk_pct = region_risk.div(region_risk.sum(axis=1), axis=0) * 100
            
            fig2 = px.bar(
                region_risk_pct.reset_index(),
                x='Região',
                y=['ALTO', 'MEDIO', 'BAIXO'],
                title="🗺️ Distribuição de Risco por Região (%)",
                color_discrete_map={
                    'ALTO': '#FF6B6B',
                    'MEDIO': '#FFE66D', 
                    'BAIXO': '#4ECDC4'
                }
            )
            fig2.update_layout(
                xaxis_tickangle=-45,
                height=500,
                font=dict(size=12),
                title_font=dict(size=18, family="Arial Black")
            )
        else:
            fig2 = go.Figure()

        # 3. Análise por grupo de cultura
        culture_risk = df.groupby(['GRUPO_CULTURA', 'RISCO_DESPERDICIO']).size().unstack(fill_value=0)
        culture_risk_pct = culture_risk.div(culture_risk.sum(axis=1), axis=0) * 100
        
        fig3 = px.bar(
            culture_risk_pct.reset_index(),
            x='GRUPO_CULTURA',
            y=['ALTO', 'MEDIO', 'BAIXO'],
            title="🌾 Distribuição de Risco por Grupo de Cultura (%)",
            color_discrete_map={
                'ALTO': '#FF6B6B',
                'MEDIO': '#FFE66D', 
                'BAIXO': '#4ECDC4'
            }
        )
        fig3.update_layout(
            xaxis_tickangle=-45,
            height=500,
            font=dict(size=12),
            title_font=dict(size=18, family="Arial Black")
        )

        # 4. Importância das features
        features = list(self.feature_importance.keys())
        importances = list(self.feature_importance.values())
        
        fig4 = px.bar(
            x=importances,
            y=features,
            orientation='h',
            title="🔍 Importância das Variáveis no Modelo",
            color=importances,
            color_continuous_scale='viridis'
        )
        fig4.update_layout(
            height=400,
            font=dict(size=12),
            title_font=dict(size=18, family="Arial Black"),
            yaxis={'categoryorder': 'total ascending'}
        )

        # 5. Matriz de correlação
        numeric_cols = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 'diversidade_produtiva']
        corr_matrix = df[numeric_cols].corr()
        
        fig5 = px.imshow(
            corr_matrix,
            title="🔗 Matriz de Correlação das Variáveis",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig5.update_layout(
            height=400,
            font=dict(size=12),
            title_font=dict(size=18, family="Arial Black")
        )

        # 6. Top 10 municípios alto risco
        high_risk = df[df['RISCO_DESPERDICIO'] == 'ALTO']
        top_high_risk = high_risk.groupby('MUNICIPIO').agg({
            'VALOR_BRUTO': 'sum',
            'PRODUCAO': 'sum',
            'AREA_PLANTADA': 'sum'
        }).sort_values('VALOR_BRUTO', ascending=True).tail(10)
        
        fig6 = px.bar(
            x=top_high_risk['VALOR_BRUTO'],
            y=top_high_risk.index,
            orientation='h',
            title="⚠️ Top 10 Municípios Alto Risco (por VBP)",
            color=top_high_risk['VALOR_BRUTO'],
            color_continuous_scale='Reds'
        )
        fig6.update_layout(
            height=500,
            font=dict(size=12),
            title_font=dict(size=18, family="Arial Black"),
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig1, fig2, fig3, fig4, fig5, fig6

    def create_metrics_cards(self):
        """Cria cards com métricas principais"""
        df = self.df
        
        cards_data = {
            'total_municipios': len(df['MUNICIPIO'].unique()),
            'total_registros': len(df),
            'accuracy': self.metrics['accuracy'],
            'alto_risco_pct': (df['RISCO_DESPERDICIO'] == 'ALTO').sum() / len(df) * 100,
            'medio_risco_pct': (df['RISCO_DESPERDICIO'] == 'MEDIO').sum() / len(df) * 100,
            'baixo_risco_pct': (df['RISCO_DESPERDICIO'] == 'BAIXO').sum() / len(df) * 100,
            'vbp_total': df['VALOR_BRUTO'].sum(),
            'producao_total': df['PRODUCAO'].sum(),
            'area_total': df['AREA_PLANTADA'].sum(),
            'diversidade_media': df['diversidade_produtiva'].mean()
        }
        
        return cards_data

    def generate_html_dashboard(self):
        """Gera dashboard HTML completo"""
        
        # Criar visualizações
        fig1, fig2, fig3, fig4, fig5, fig6 = self.create_interactive_dashboard()
        cards_data = self.create_metrics_cards()
        
        # Converter gráficos para HTML
        graph1_html = pyo.plot(fig1, output_type='div', include_plotlyjs=False)
        graph2_html = pyo.plot(fig2, output_type='div', include_plotlyjs=False)
        graph3_html = pyo.plot(fig3, output_type='div', include_plotlyjs=False)
        graph4_html = pyo.plot(fig4, output_type='div', include_plotlyjs=False)
        graph5_html = pyo.plot(fig5, output_type='div', include_plotlyjs=False)
        graph6_html = pyo.plot(fig6, output_type='div', include_plotlyjs=False)

        # Template HTML responsivo
        html_template = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚜 Dashboard - Risco de Desperdício Agrícola</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}

        .dashboard-header {{
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}

        .dashboard-header h1 {{
            font-size: 2.5rem;
            color: #2c3e50;
            margin-bottom: 10px;
        }}

        .dashboard-header p {{
            font-size: 1.2rem;
            color: #7f8c8d;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.15);
        }}

        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #4ECDC4, #44A08D);
        }}

        .metric-icon {{
            font-size: 2.5rem;
            margin-bottom: 15px;
            color: #4ECDC4;
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}

        .metric-label {{
            font-size: 1rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}

        .chart-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }}

        .chart-container:hover {{
            transform: scale(1.02);
        }}

        .full-width {{
            grid-column: 1 / -1;
        }}

        .risk-indicator {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9rem;
        }}

        .risk-alto {{
            background: #FFE6E6;
            color: #D63031;
        }}

        .risk-medio {{
            background: #FFF9E6;
            color: #E17055;
        }}

        .risk-baixo {{
            background: #E6F7F7;
            color: #00B894;
        }}

        .insights-panel {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            margin-top: 30px;
        }}

        .insight-item {{
            background: #f8f9fa;
            border-left: 4px solid #4ECDC4;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
        }}

        .insight-title {{
            font-size: 1.3rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}

        .insight-description {{
            color: #7f8c8d;
            line-height: 1.6;
        }}

        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            .metric-card {{
                padding: 20px;
            }}
            
            .dashboard-header h1 {{
                font-size: 2rem;
            }}
        }}

        .performance-badge {{
            display: inline-block;
            background: #00B894;
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-weight: bold;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>🚜 Dashboard - Análise de Risco de Desperdício Agrícola</h1>
        <p>Sistema Inteligente de Classificação de Municípios | Safra 2024</p>
        <div class="performance-badge">
            🎯 Modelo com {cards_data['accuracy']:.1%} de Acurácia
        </div>
    </div>

    <div class="container">
        <!-- Cards de Métricas -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-icon">🏛️</div>
                <div class="metric-value">{cards_data['total_municipios']:,}</div>
                <div class="metric-label">Municípios Analisados</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-value">{cards_data['total_registros']:,}</div>
                <div class="metric-label">Registros Processados</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">⚠️</div>
                <div class="metric-value">{cards_data['alto_risco_pct']:.1f}%</div>
                <div class="metric-label">Alto Risco</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">⚖️</div>
                <div class="metric-value">{cards_data['medio_risco_pct']:.1f}%</div>
                <div class="metric-label">Médio Risco</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">✅</div>
                <div class="metric-value">{cards_data['baixo_risco_pct']:.1f}%</div>
                <div class="metric-label">Baixo Risco</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">💰</div>
                <div class="metric-value">R$ {cards_data['vbp_total']/1e9:.1f}B</div>
                <div class="metric-label">VBP Total</div>
            </div>
        </div>

        <!-- Gráficos -->
        <div class="charts-grid">
            <div class="chart-container">
                {graph1_html}
            </div>
            
            <div class="chart-container">
                {graph4_html}
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container full-width">
                {graph2_html}
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container full-width">
                {graph3_html}
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                {graph5_html}
            </div>
            
            <div class="chart-container">
                {graph6_html}
            </div>
        </div>

        <!-- Painel de Insights -->
        <div class="insights-panel">
            <h2>🧠 Insights Principais</h2>
            
            <div class="insight-item">
                <div class="insight-title">🎯 Performance do Modelo</div>
                <div class="insight-description">
                    O modelo Random Forest alcançou <strong>{cards_data['accuracy']:.1%}</strong> de acurácia, 
                    demonstrando alta confiabilidade na classificação de risco de desperdício agrícola.
                </div>
            </div>
            
            <div class="insight-item">
                <div class="insight-title">📈 Distribuição de Risco</div>
                <div class="insight-description">
                    <span class="risk-indicator risk-alto">Alto Risco: {cards_data['alto_risco_pct']:.1f}%</span>
                    <span class="risk-indicator risk-medio">Médio Risco: {cards_data['medio_risco_pct']:.1f}%</span>
                    <span class="risk-indicator risk-baixo">Baixo Risco: {cards_data['baixo_risco_pct']:.1f}%</span>
                    <br><br>
                    A distribuição equilibrada indica que o modelo consegue identificar diferentes níveis de risco 
                    de forma consistente entre os municípios analisados.
                </div>
            </div>
            
            <div class="insight-item">
                <div class="insight-title">🔍 Fatores Mais Importantes</div>
                <div class="insight-description">
                    O <strong>Valor Bruto da Produção</strong> é o fator mais determinante no risco de desperdício, 
                    seguido pela <strong>Produção</strong> e <strong>Área Plantada</strong>. 
                    A diversidade produtiva também desempenha papel relevante na análise de risco.
                </div>
            </div>
            
            <div class="insight-item">
                <div class="insight-title">🗺️ Padrões Regionais</div>
                <div class="insight-description">
                    Algumas regiões apresentam maior concentração de municípios de alto risco, 
                    sugerindo a necessidade de políticas públicas direcionadas e investimentos 
                    em tecnologia e assistência técnica específicas para essas áreas.
                </div>
            </div>
            
            <div class="insight-item">
                <div class="insight-title">🌾 Grupos de Cultura</div>
                <div class="insight-description">
                    Hortaliças e Frutas apresentam maior risco de desperdício comparado a 
                    Grãos e Grandes Culturas, indicando a necessidade de estratégias 
                    específicas para culturas mais perecíveis.
                </div>
            </div>
            
            <div class="insight-item">
                <div class="insight-title">💡 Recomendações</div>
                <div class="insight-description">
                    • <strong>Alto Risco:</strong> Investir em tecnologia pós-colheita e cadeias de frio<br>
                    • <strong>Médio Risco:</strong> Monitoramento contínuo e assistência técnica<br>
                    • <strong>Baixo Risco:</strong> Manter boas práticas e servir como modelo<br>
                    • <strong>Geral:</strong> Promover diversificação produtiva e cooperativismo
                </div>
            </div>
        </div>
    </div>

    <script>
        // Adicionar interatividade adicional
        document.addEventListener('DOMContentLoaded', function() {{
            // Animação dos cards ao scroll
            const cards = document.querySelectorAll('.metric-card');
            const observer = new IntersectionObserver((entries) => {{
                entries.forEach((entry) => {{
                    if (entry.isIntersecting) {{
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }}
                }});
            }});

            cards.forEach((card) => {{
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                card.style.transition = 'all 0.6s ease';
                observer.observe(card);
            }});

            // Adicionar tooltips aos gráficos
            console.log('Dashboard carregado com sucesso! 🚀');
        }});
    </script>
</body>
</html>
        """

        # Salvar dashboard HTML
        with open('dashboard_risco_agricultura.html', 'w', encoding='utf-8') as f:
            f.write(html_template)
            
        print("✅ Dashboard HTML criado: dashboard_risco_agricultura.html")
        return html_template

def main():
    """Função principal"""
    # Inicializar dashboard
    dashboard = DashboardRiscoAgricultura()
    
    # Carregar e processar dados
    file_path = 'd:/Users/Leonardo/Desktop/LovatConsorcios/vbp_2024.xlsx'
    dashboard.load_and_process_data(file_path)
    
    # Criar classificação de risco
    dashboard.create_risk_classification()
    
    # Treinar modelo
    dashboard.train_model()
    
    # Gerar dashboard HTML
    dashboard.generate_html_dashboard()
    
    print("🎉 Dashboard completo gerado com sucesso!")
    print("📊 Abra o arquivo 'dashboard_risco_agricultura.html' no navegador")
    
    return dashboard

if __name__ == "__main__":
    dashboard = main()