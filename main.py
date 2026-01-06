#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script Principal - Classificador de Risco Agrícola
Ponto de entrada principal do sistema com arquitetura MVC
"""

import os
import sys
import subprocess

def main():
    """Função principal do sistema"""
    
    print("=" * 80)
    print(" " * 15 + "CLASSIFICADOR DE RISCO DE DESPERDÍCIO AGRÍCOLA")
    print(" " * 25 + "Sistema com Arquitetura MVC")
    print("=" * 80)
    
    print("\n📁 Estrutura do Projeto:")
    print("   models/       - Lógica de ML e processamento de dados")
    print("   views/        - Dashboard interativo (HTML)")
    print("   controllers/  - Scripts de controle e execução")
    print("   data/         - Dados de entrada e resultados")
    print("   outputs/      - Gráficos e visualizações")
    print("   notebooks/    - Notebooks Jupyter das aulas")
    print("   docs/         - Documentação completa")
    
    print("\n🚀 Opções de Execução:")
    print("   1. Executar Análise Completa de ML")
    print("   2. Abrir Dashboard Interativo")
    print("   3. Ver Documentação")
    print("   4. Verificar Estrutura de Arquivos")
    print("   0. Sair")
    
    escolha = input("\n👉 Escolha uma opção: ")
    
    if escolha == "1":
        print("\n🔄 Executando análise de Machine Learning...")
        print("   Aguarde, isto pode levar alguns minutos...\n")
        subprocess.run([sys.executable, "models/analise_rapida.py"])
        
        print("\n✅ Análise concluída!")
        print("📊 Resultados salvos em: data/resultados_ml.json")
        print("📈 Gráficos salvos em: outputs/")
        
        abrir_dash = input("\n❓ Deseja abrir o dashboard? (s/n): ")
        if abrir_dash.lower() == 's':
            abrir_dashboard()
    
    elif escolha == "2":
        abrir_dashboard()
    
    elif escolha == "3":
        print("\n📚 Documentação disponível:")
        print("   - README.md         : Visão geral do projeto")
        print("   - docs/README_ML.md : Documentação técnica completa")
        print("   - docs/GUIA_RAPIDO.md : Guia rápido de uso")
        
        if os.path.exists("README.md"):
            abrir = input("\n❓ Abrir README.md? (s/n): ")
            if abrir.lower() == 's':
                if sys.platform == 'win32':
                    os.startfile("README.md")
                else:
                    subprocess.run(["open", "README.md"])
    
    elif escolha == "4":
        verificar_estrutura()
    
    elif escolha == "0":
        print("\n👋 Encerrando...")
        sys.exit(0)
    
    else:
        print("\n❌ Opção inválida!")

def abrir_dashboard():
    """Abre o dashboard interativo no navegador"""
    dashboard_path = "views/dashboard_final.html"
    
    if not os.path.exists(dashboard_path):
        print(f"\n❌ Erro: Dashboard não encontrado em {dashboard_path}")
        print("   Execute primeiro a análise (opção 1)")
        return
    
    print(f"\n🌐 Abrindo dashboard: {dashboard_path}")
    
    if sys.platform == 'win32':
        os.startfile(dashboard_path)
    elif sys.platform == 'darwin':  # macOS
        subprocess.run(["open", dashboard_path])
    else:  # Linux
        subprocess.run(["xdg-open", dashboard_path])
    
    print("✅ Dashboard aberto no navegador!")

def verificar_estrutura():
    """Verifica a estrutura de arquivos do projeto"""
    print("\n📂 Verificando estrutura do projeto...\n")
    
    estrutura = {
        "models": ["analise_rapida.py"],
        "views": ["dashboard_final.html"],
        "controllers": ["executar_analise.py"],
        "data": ["resultados_ml.json", "comparacao_modelos.csv"],
        "outputs": ["comparacao_metricas.png", "feature_importance.png"],
        "notebooks": ["Aula"],
        "docs": ["README_ML.md", "GUIA_RAPIDO.md"]
    }
    
    for pasta, arquivos in estrutura.items():
        status = "✅" if os.path.exists(pasta) else "❌"
        print(f"{status} {pasta}/")
        
        if os.path.exists(pasta):
            for arquivo in arquivos:
                caminho = os.path.join(pasta, arquivo)
                status_arq = "  ✓" if os.path.exists(caminho) else "  ✗"
                print(f"{status_arq} {arquivo}")
    
    # Verificar dados brutos
    print("\n📊 Dados VBP:")
    data_files = [f for f in os.listdir("data") if f.startswith(("VBP", "vbp"))]
    print(f"   ✓ {len(data_files)} arquivos VBP encontrados")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Operação cancelada pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)
