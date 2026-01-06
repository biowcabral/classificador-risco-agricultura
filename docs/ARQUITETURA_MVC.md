# 🏗️ Arquitetura MVC do Sistema

## Visão Geral

Este projeto foi reestruturado seguindo o padrão **MVC (Model-View-Controller)** para melhor organização, manutenibilidade e escalabilidade do código.

## 📂 Estrutura de Diretórios

```
classificador-risco-agricultura/
│
├── 📄 main.py                          # Ponto de entrada principal do sistema
├── 📄 README.md                         # Documentação principal
├── 📄 .gitignore                        # Arquivos ignorados pelo Git
│
├── 📁 models/                           # MODEL - Lógica de Negócio
│   └── analise_rapida.py               # Core do ML: análise e treinamento
│
├── 📁 views/                            # VIEW - Interface do Usuário
│   └── dashboard_final.html            # Dashboard interativo com Chart.js
│
├── 📁 controllers/                      # CONTROLLER - Orquestração
│   └── executar_analise.py             # Menu e controle de execução
│
├── 📁 data/                             # Dados (Input/Output)
│   ├── VBP*.xls, vbp*.xlsx             # Dados brutos (13 anos)
│   ├── resultados_ml.json              # Resultados da análise
│   └── comparacao_modelos.csv          # Tabela comparativa
│
├── 📁 outputs/                          # Saídas Visuais
│   ├── comparacao_metricas.png         # Gráfico de métricas
│   ├── confusion_matrix_melhor.png     # Matriz de confusão
│   ├── feature_importance.png          # Importância de features
│   └── evolucao_temporal.png           # Análise temporal
│
├── 📁 notebooks/                        # Notebooks Jupyter
│   └── Aula/                           # 11 notebooks das aulas
│
├── 📁 docs/                             # Documentação
│   ├── README_ML.md                    # Doc técnica completa
│   ├── GUIA_RAPIDO.md                  # Quick start
│   └── ARQUITETURA_MVC.md              # Este arquivo
│
└── 📁 obsoletos/                        # Arquivos Legados
    ├── analise_temporal_agricultura*.py
    ├── dashboard_*.html (versões antigas)
    └── scripts auxiliares não utilizados
```

## 🎯 Padrão MVC Aplicado

### 📊 MODEL (models/)

**Responsabilidade:** Lógica de negócio, processamento de dados e Machine Learning

**Arquivo Principal:** `analise_rapida.py`

**Funções:**
- Carregamento de dados VBP multi-anos
- Pré-processamento e limpeza
- Engenharia de features
- Treinamento de 7 modelos ML
- Feature selection (SelectKBest, Feature Importance)
- Geração de métricas e resultados

**Saídas:**
- `data/resultados_ml.json` - Resultados completos
- `data/comparacao_modelos.csv` - Tabela comparativa
- `outputs/*.png` - Visualizações

### 🖥️ VIEW (views/)

**Responsabilidade:** Interface do usuário e visualização de dados

**Arquivo Principal:** `dashboard_final.html`

**Características:**
- Dashboard interativo HTML/CSS/JavaScript
- Chart.js para gráficos dinâmicos
- 4 abas: Visão Geral, Modelos, Features, Análise Detalhada
- Carrega dados de `../data/resultados_ml.json`
- Responsivo e moderno

**Visualizações:**
- Comparação de accuracy
- Tabela de modelos
- Feature importance
- Análise detalhada de cada modelo

### 🎮 CONTROLLER (controllers/)

**Responsabilidade:** Controle de fluxo e orquestração entre Model e View

**Arquivo Principal:** `executar_analise.py`

**Funções:**
- Menu interativo
- Execução do modelo (chama `models/analise_rapida.py`)
- Abertura do dashboard (chama `views/dashboard_final.html`)
- Validação de dependências

## 🔄 Fluxo de Execução

```
┌─────────────────┐
│   main.py       │  ← Ponto de entrada
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  controllers/               │
│  executar_analise.py        │  ← Controla fluxo
└────────┬────────────────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌──────────────────┐
│   models/       │  │   views/         │
│ analise_rapida  │  │ dashboard_final  │
└────────┬────────┘  └────────┬─────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────┐
│   data/         outputs/            │
│   *.json        *.png               │
│   *.csv                             │
└─────────────────────────────────────┘
```

## 📥 Fluxo de Dados

1. **Input:** `data/VBP*.xls` (dados brutos)
2. **Processing:** `models/analise_rapida.py` processa
3. **Output:** Gera `data/*.json`, `data/*.csv`, `outputs/*.png`
4. **Visualization:** `views/dashboard_final.html` lê resultados
5. **Control:** `controllers/executar_analise.py` orquestra tudo

## 🚀 Como Executar

### Método 1: Script Principal (Recomendado)

```bash
python main.py
```

Menu interativo com opções:
1. Executar análise completa
2. Abrir dashboard
3. Ver documentação
4. Verificar estrutura

### Método 2: Execução Direta

```bash
# Model: Executar análise
cd models
python analise_rapida.py

# View: Abrir dashboard
start ../views/dashboard_final.html
```

### Método 3: Controller

```bash
python controllers/executar_analise.py
```

## 🔧 Dependências entre Componentes

```
MODEL ──────► DATA ──────► VIEW
  │                          ▲
  │                          │
  └──────► OUTPUTS ──────────┘
              ▲
              │
         CONTROLLER
```

- **Model** gera dados e outputs
- **View** consome dados e outputs
- **Controller** orquestra Model e View
- **Data/Outputs** são camadas de persistência

## 📝 Boas Práticas Implementadas

### ✅ Separação de Responsabilidades
- Lógica ML isolada em `models/`
- Interface isolada em `views/`
- Controle isolado em `controllers/`

### ✅ Caminhos Relativos
- Model usa `../data/` e `../outputs/`
- View usa `../data/`
- Funciona independente do diretório de execução

### ✅ Dados Separados do Código
- Dados brutos em `data/`
- Resultados em `data/` e `outputs/`
- Versionamento seletivo (.gitignore)

### ✅ Documentação Centralizada
- `docs/` contém toda documentação técnica
- README.md no root para visão geral
- Cada componente é auto-documentado

### ✅ Arquivos Obsoletos Isolados
- `obsoletos/` contém código legado
- Não afeta funcionamento atual
- Mantido para referência histórica

## 🔄 Vantagens da Arquitetura MVC

1. **Manutenibilidade**
   - Fácil localizar e modificar funcionalidades
   - Cada pasta tem responsabilidade clara

2. **Escalabilidade**
   - Adicionar novos modelos: apenas `models/`
   - Adicionar novos dashboards: apenas `views/`
   - Adicionar novos controllers: apenas `controllers/`

3. **Testabilidade**
   - Testar Model independentemente da View
   - Testar View com dados mockados
   - Testar Controller isoladamente

4. **Colaboração**
   - Data Scientist trabalha em `models/`
   - Frontend Developer trabalha em `views/`
   - DevOps trabalha em `controllers/`
   - Sem conflitos de merge

5. **Reusabilidade**
   - Model pode ser usado por outros sistemas
   - View pode conectar a outros backends
   - Controller pode orquestrar diferentes Models/Views

## 🆚 Antes vs Depois

### ❌ Antes (Estrutura Plana)
```
projeto/
├── analise_temporal_agricultura.py
├── analise_temporal_agricultura_backup.py
├── analise_temporal_agricultura_completa.py
├── analise_rapida.py
├── dashboard_final.html
├── dashboard_ml_comparativo.html
├── dashboard_interativo.html
├── resultados_ml.json
├── comparacao_modelos.csv
├── *.png (misturado)
└── 20+ arquivos sem organização
```

**Problemas:**
- Difícil encontrar arquivos
- Muitos arquivos obsoletos misturados
- Sem separação de responsabilidades
- Caminhos hardcoded

### ✅ Depois (Arquitetura MVC)
```
projeto/
├── main.py (ponto de entrada)
├── models/ (lógica ML)
├── views/ (interface)
├── controllers/ (orquestração)
├── data/ (dados)
├── outputs/ (resultados)
├── notebooks/ (análises)
├── docs/ (documentação)
└── obsoletos/ (legado isolado)
```

**Vantagens:**
- Estrutura clara e organizada
- Fácil navegação
- Separação de responsabilidades
- Caminhos relativos consistentes
- Obsoletos isolados

## 📚 Referências

- **MVC Pattern:** https://en.wikipedia.org/wiki/Model–view–controller
- **Clean Architecture:** Robert C. Martin
- **Python Project Structure:** https://docs.python-guide.org/writing/structure/

---

**Atualizado:** Janeiro 2026
