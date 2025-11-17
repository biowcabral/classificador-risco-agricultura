# 🚜 Sistema de Classificação de Risco de Desperdício Agrícola

## 📋 Descrição

Sistema de Machine Learning para classificação de municípios brasileiros em categorias de risco de desperdício agrícola, baseado em dados do Valor Bruto da Produção (VBP) 2024. O sistema utiliza algoritmo Random Forest para categorizar municípios em três níveis: **BAIXO**, **MÉDIO** e **ALTO** risco.

## 🎯 Objetivos

- Identificar municípios com maior probabilidade de desperdício agrícola
- Analisar padrões de produção, área plantada e diversidade produtiva
- Gerar visualizações interativas para tomada de decisão
- Fornecer insights para políticas públicas agrícolas

## 📊 Funcionalidades

### 🔍 Análise de Dados
- **Diversidade Produtiva**: Cálculo do número de culturas por município
- **Classificação de Risco**: Algoritmo baseado em quantis de produção
- **Correlação de Variáveis**: Análise de relações entre indicadores

### 📈 Visualizações
- Dashboard interativo com Plotly
- Distribuição de risco por município
- Análise por grupo de cultura
- Matriz de correlação
- Ranking de municípios de alto risco
- Importância das variáveis no modelo

### 🤖 Machine Learning
- **Algoritmo**: Random Forest Classifier
- **Features**: Produção, área plantada, VBP, diversidade produtiva
- **Métricas**: Acurácia, precisão, recall, F1-score
- **Validação**: Train/test split com estratificação

## 🛠️ Tecnologias Utilizadas

```
Python 3.13+
├── pandas - Manipulação de dados
├── numpy - Operações numéricas
├── scikit-learn - Machine Learning
├── plotly - Visualizações interativas
├── matplotlib - Gráficos estáticos
└── seaborn - Visualizações estatísticas
```

## 📂 Estrutura do Projeto

```
├── municipio_food_waste_risk_classifier.py     # Classificador básico
├── municipio_food_waste_risk_classifier_detailed.py  # Versão detalhada
├── dashboard_risco_agricultura.py              # Dashboard interativo
├── DOCUMENTACAO_CLASSIFICADOR_RISCO.md         # Documentação técnica
├── .gitignore                                  # Arquivos ignorados pelo Git
└── README.md                                   # Este arquivo
```

## 🚀 Como Usar

### 1. Instalação de Dependências
```bash
pip install pandas numpy scikit-learn plotly matplotlib seaborn openpyxl
```

### 2. Executar Classificador Básico
```bash
python municipio_food_waste_risk_classifier.py
```

### 3. Gerar Dashboard Interativo
```bash
python dashboard_risco_agricultura.py
```

O dashboard será salvo como `dashboard_risco_agricultura.html` e pode ser aberto em qualquer navegador.

## 📊 Dados Necessários

O sistema requer um arquivo Excel (`vbp_2024.xlsx`) com as seguintes colunas:
- **Município**: Nome do município
- **Produção**: Volume de produção
- **Área (ha)**: Área plantada em hectares
- **VBP**: Valor Bruto da Produção
- **Grupo**: Grupo da cultura
- **Cultura**: Tipo de cultura

## 🔬 Metodologia

### Classificação de Risco
O algoritmo classifica municípios baseado em score calculado através de quantis:

- **Quantil inferior (33%)**: +1 ponto (risco)
- **Quantil superior (66%)**: -1 ponto (proteção)
- **Score ≥ 2**: ALTO risco
- **Score ≤ -2**: BAIXO risco
- **-1 < Score < 2**: MÉDIO risco

### Variáveis Analisadas
1. **Produção**: Volume total produzido
2. **Área Plantada**: Extensão cultivada
3. **Valor Bruto**: Valor econômico da produção
4. **Diversidade Produtiva**: Número de culturas diferentes
5. **Grupo de Cultura**: Categoria da cultura

## 📈 Resultados Esperados

- **Acurácia do Modelo**: ~85-90%
- **Dashboard Interativo**: Visualizações em tempo real
- **Relatórios**: Análises detalhadas por região e cultura
- **Rankings**: Top municípios por categoria de risco

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👥 Autores

- **Rayanne** - *Pesquisa de Mestrado*
- **Leonardo** - *Desenvolvimento e Implementação*

## 📞 Contato

Para dúvidas ou sugestões, entre em contato através dos issues do GitHub.

---

🌾 *"Tecnologia a serviço da agricultura sustentável"* 🌾