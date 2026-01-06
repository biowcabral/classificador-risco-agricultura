"""
Script de Execução Rápida - Sistema de Análise ML Agrícola
Facilita a execução e visualização dos resultados
"""

import subprocess
import webbrowser
import os
import sys
from pathlib import Path

def print_header(text):
    """Imprime cabeçalho formatado"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_info(text):
    """Imprime informação"""
    print(f"ℹ️  {text}")

def print_success(text):
    """Imprime sucesso"""
    print(f"✅ {text}")

def print_error(text):
    """Imprime erro"""
    print(f"❌ {text}")

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    print_header("VERIFICANDO DEPENDÊNCIAS")
    
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
    }
    
    optional = {
        'xgboost': 'xgboost',
        'shap': 'shap',
        'mlxtend': 'mlxtend'
    }
    
    missing_required = []
    missing_optional = []
    
    # Verificar dependências obrigatórias
    for module, package in required.items():
        try:
            __import__(module)
            print_success(f"{package} instalado")
        except ImportError:
            print_error(f"{package} NÃO instalado (OBRIGATÓRIO)")
            missing_required.append(package)
    
    # Verificar dependências opcionais
    for module, package in optional.items():
        try:
            __import__(module)
            print_success(f"{package} instalado (opcional)")
        except ImportError:
            print_info(f"{package} NÃO instalado (opcional - funcionalidades limitadas)")
            missing_optional.append(package)
    
    if missing_required:
        print_error("\nDependências obrigatórias ausentes!")
        print_info(f"Execute: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print_info("\nDependências opcionais ausentes - algumas funcionalidades estarão desabilitadas")
        print_info(f"Para instalar: pip install {' '.join(missing_optional)}")
    
    print_success("\n✅ Todas as dependências obrigatórias estão instaladas!")
    return True

def check_data_files():
    """Verifica se os arquivos de dados existem"""
    print_header("VERIFICANDO ARQUIVOS DE DADOS")
    
    vbp_files = list(Path('.').glob('VBP*.xls*')) + list(Path('.').glob('vbp*.xls*'))
    
    if not vbp_files:
        print_error("Nenhum arquivo VBP encontrado!")
        print_info("Certifique-se de que os arquivos VBP*.xlsx estão na pasta atual")
        return False
    
    print_success(f"Encontrados {len(vbp_files)} arquivo(s) VBP:")
    for f in vbp_files:
        print(f"   • {f.name}")
    
    return True

def run_analysis():
    """Executa a análise completa"""
    print_header("EXECUTANDO ANÁLISE COMPLETA")
    print_info("Isso pode levar alguns minutos...")
    
    try:
        result = subprocess.run(
            [sys.executable, 'analise_temporal_agricultura_completa.py'],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print_success("\n✅ Análise concluída com sucesso!")
            return True
        else:
            print_error("\n❌ Erro durante a execução da análise")
            return False
            
    except FileNotFoundError:
        print_error("Arquivo analise_temporal_agricultura_completa.py não encontrado!")
        return False
    except Exception as e:
        print_error(f"Erro ao executar análise: {e}")
        return False

def open_dashboard():
    """Abre o dashboard no navegador"""
    print_header("ABRINDO DASHBOARD INTERATIVO")
    
    dashboard_file = Path('dashboard_ml_comparativo.html')
    
    if not dashboard_file.exists():
        print_error("Arquivo dashboard_ml_comparativo.html não encontrado!")
        return False
    
    try:
        webbrowser.open(f'file://{dashboard_file.absolute()}')
        print_success("Dashboard aberto no navegador!")
        return True
    except Exception as e:
        print_error(f"Erro ao abrir dashboard: {e}")
        print_info(f"Abra manualmente: {dashboard_file.absolute()}")
        return False

def list_generated_files():
    """Lista arquivos gerados pela análise"""
    print_header("ARQUIVOS GERADOS")
    
    expected_files = [
        'feature_importance.png',
        'comparacao_metricas.png',
        'comparacao_cv.png',
        'confusion_matrix_melhor.png',
        'clustering_comparison.png',
        'elbow_method.png',
        'evolucao_temporal.png',
        'comparacao_modelos.csv',
        'resultados_ml.json'
    ]
    
    found = []
    missing = []
    
    for filename in expected_files:
        if Path(filename).exists():
            found.append(filename)
            print_success(filename)
        else:
            missing.append(filename)
            print_info(f"{filename} (não gerado ainda)")
    
    print(f"\n   Total: {len(found)}/{len(expected_files)} arquivos encontrados")
    
    return len(found) > 0

def show_menu():
    """Mostra menu interativo"""
    while True:
        print_header("SISTEMA DE ANÁLISE ML AGRÍCOLA")
        print("\n  Escolha uma opção:")
        print("  1. ✅ Verificar dependências")
        print("  2. 📂 Verificar arquivos de dados")
        print("  3. 🚀 Executar análise completa")
        print("  4. 📊 Abrir dashboard interativo")
        print("  5. 📁 Listar arquivos gerados")
        print("  6. 🔄 Executar tudo (verificar + analisar + abrir)")
        print("  7. 📖 Abrir documentação (README)")
        print("  0. ❌ Sair")
        print("\n" + "=" * 80)
        
        choice = input("\n  Digite sua escolha: ").strip()
        
        if choice == '1':
            check_dependencies()
            input("\nPressione Enter para continuar...")
            
        elif choice == '2':
            check_data_files()
            input("\nPressione Enter para continuar...")
            
        elif choice == '3':
            if check_dependencies() and check_data_files():
                run_analysis()
            input("\nPressione Enter para continuar...")
            
        elif choice == '4':
            open_dashboard()
            input("\nPressione Enter para continuar...")
            
        elif choice == '5':
            list_generated_files()
            input("\nPressione Enter para continuar...")
            
        elif choice == '6':
            print_header("EXECUÇÃO COMPLETA")
            if check_dependencies() and check_data_files():
                if run_analysis():
                    list_generated_files()
                    open_dashboard()
            input("\nPressione Enter para continuar...")
            
        elif choice == '7':
            readme = Path('README_ML.md')
            if readme.exists():
                try:
                    webbrowser.open(f'file://{readme.absolute()}')
                    print_success("Documentação aberta!")
                except:
                    print_info(f"Abra manualmente: {readme.absolute()}")
            else:
                print_error("README_ML.md não encontrado!")
            input("\nPressione Enter para continuar...")
            
        elif choice == '0':
            print_header("ATÉ LOGO!")
            break
            
        else:
            print_error("Opção inválida!")
            input("\nPressione Enter para continuar...")

def main():
    """Função principal"""
    print("\n")
    print("=" * 80)
    print("  🌾 SISTEMA DE ANÁLISE TEMPORAL DE RISCO AGRÍCOLA 🌾")
    print("  Comparação Abrangente de Metodologias de Machine Learning")
    print("=" * 80)
    print("\n  Baseado nas Aulas 1-9:")
    print("  • Iris, Diabetes, Predictive Analytics")
    print("  • Machine Failure, Churn Prediction")
    print("  • Feature Selection, Wine Clustering")
    print("  • Health Ageing, Obesity, Breast Cancer XAI")
    print("  • Groceries (Association Rules)")
    print("=" * 80)
    
    # Se argumentos foram passados
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'check':
            check_dependencies()
            check_data_files()
        elif command == 'run':
            if check_dependencies() and check_data_files():
                run_analysis()
        elif command == 'dashboard':
            open_dashboard()
        elif command == 'all':
            if check_dependencies() and check_data_files():
                if run_analysis():
                    open_dashboard()
        else:
            print_error(f"Comando desconhecido: {command}")
            print_info("Comandos disponíveis: check, run, dashboard, all")
    else:
        # Modo interativo
        show_menu()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("  ⚠️  Execução interrompida pelo usuário")
        print("=" * 80)
        sys.exit(0)
    except Exception as e:
        print_error(f"Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
