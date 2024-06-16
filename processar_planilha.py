import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import sys
import os
import logging

# Configuração do logging
logging.basicConfig(filename='processamento_de_dados.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def ler_dados(input_file):
    try:
        file_extension = os.path.splitext(input_file)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(input_file)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(input_file)
        else:
            raise ValueError("Formato de arquivo não suportado: " + file_extension)
        
        logging.info(f"Arquivo {input_file} lido com sucesso.")
        print("Colunas disponíveis:")
        print(df.columns.tolist())  # Lista as colunas disponíveis
        print("Tipos de dados das colunas:")
        print(df.dtypes)  # Verifica os tipos de dados das colunas
        print("Dados originais:")
        print(df.head())
        return df
    except FileNotFoundError:
        logging.error(f"Erro: Arquivo {input_file} não encontrado.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo: {e}")
        sys.exit(1)

def limpeza_dados(df):
    df_cleaned = df.dropna(subset=['VIN (1-10)', 'DOL Vehicle ID'])
    df_cleaned['County'] = df_cleaned['County'].fillna('Valor_Padrão')
    logging.info("Dados limpos com sucesso.")
    print("Dados após limpeza:")
    print(df_cleaned.head())
    return df_cleaned

def normalizar_dados(df):
    df['DOL Vehicle ID'] = (df['DOL Vehicle ID'] - df['DOL Vehicle ID'].min()) / (df['DOL Vehicle ID'].max() - df['DOL Vehicle ID'].min())
    logging.info("Dados normalizados com sucesso.")
    print("Dados após normalização:")
    print(df.head())
    return df

def aplicar_filtros(df):
    print("Dados antes dos filtros:")
    print(df.head())
    df = df[df['DOL Vehicle ID'] < 10]
    logging.info("Filtros aplicados com sucesso.")
    print("Dados após aplicação de filtros:")
    print(df.head())
    return df

def transformar_dados(df):
    df['E'] = df['DOL Vehicle ID'].apply(lambda x: np.log(x) if x > 0 else None)
    logging.info("Dados transformados com sucesso.")
    print("Dados após transformações avançadas:")
    print(df.head())
    return df

def analise_estatistica(df):
    df_numerico = df.select_dtypes(include=[np.number])  # Selecionar apenas colunas numéricas
    estatisticas = df_numerico.describe()
    estatisticas.loc['median'] = df_numerico.median()
    estatisticas.loc['std'] = df_numerico.std()
    logging.info("Análise estatística realizada com sucesso.")
    print("Análise estatística dos dados processados:")
    print(estatisticas)
    return estatisticas

def exportar_para_json(df, json_file):
    if os.path.exists(json_file):
        resposta = input(f"O arquivo {json_file} já existe. Deseja sobrescrever? (s/n): ")
        if resposta.lower() != 's':
            logging.info(f"Exportação para JSON cancelada: {json_file}")
            print("Exportação para JSON cancelada.")
            return
    df.to_json(json_file, orient='records', lines=True)
    logging.info(f"Dados exportados para JSON em: {json_file}")
    print(f"Dados exportados para JSON em: {json_file}")

def exportar_para_sqlite(df, sqlite_file):
    if os.path.exists(sqlite_file):
        resposta = input(f"O arquivo {sqlite_file} já existe. Deseja sobrescrever? (s/n): ")
        if resposta.lower() != 's':
            logging.info(f"Exportação para SQLite cancelada: {sqlite_file}")
            print("Exportação para SQLite cancelada.")
            return
    conn = sqlite3.connect(sqlite_file)
    df.to_sql('dados_processados', conn, if_exists='replace', index=False)
    conn.close()
    logging.info(f"Dados exportados para SQLite em: {sqlite_file}")
    print(f"Dados exportados para SQLite em: {sqlite_file}")

def gerar_relatorio(df, estatisticas, relatorio_file, output_dir):
    if os.path.exists(relatorio_file):
        resposta = input(f"O arquivo {relatorio_file} já existe. Deseja sobrescrever? (s/n): ")
        if resposta.lower() != 's':
            logging.info(f"Geração de relatório cancelada: {relatorio_file}")
            print("Geração de relatório cancelada.")
            return
    with open(relatorio_file, 'w') as f:
        f.write("Relatório de Processamento de Dados\n")
        f.write("="*50 + "\n")
        f.write("Dados Processados:\n")
        f.write(df.head().to_string() + "\n\n")
        f.write("Estatísticas Descritivas:\n")
        f.write(estatisticas.to_string() + "\n")
    logging.info(f"Relatório salvo em: {relatorio_file}")
    print(f"Relatório salvo em: {relatorio_file}")

    # Incluir gráficos no relatório
    try:
        with open(relatorio_file, 'a') as f:
            for img_file in os.listdir(output_dir):
                if img_file.endswith('.png'):
                    f.write(f"\n![{img_file}]({os.path.join(output_dir, img_file)})\n")
        logging.info("Gráficos incluídos no relatório com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao incluir gráficos no relatório: {e}")
        print(f"Erro ao incluir gráficos no relatório: {e}")

def criar_grafico_barras(df, output_dir):
    try:
        df_top_counties = df.nlargest(10, 'Media_DOL_Vehicle_ID')
        
        plt.figure(figsize=(10, 6))
        plt.bar(df_top_counties['County'], df_top_counties['Media_DOL_Vehicle_ID'])
        plt.xlabel('County')
        plt.ylabel('Media_DOL_Vehicle_ID')
        plt.title('Top 10 Condados com Maior Média de DOL Vehicle ID')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grafico_barras.png'))
        logging.info("Gráfico de barras criado e salvo com sucesso.")
        plt.show()
    except Exception as e:
        logging.error(f"Erro ao criar o gráfico de barras: {e}")
        print(f"Erro ao criar o gráfico de barras: {e}")

def criar_histograma(df, coluna, output_dir):
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(df[coluna], bins=30, edgecolor='k')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')
        plt.title(f'Histograma de {coluna}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'histograma_{coluna}.png'))
        logging.info(f"Histograma de {coluna} criado e salvo com sucesso.")
        plt.show()
    except KeyError:
        logging.error(f"Coluna '{coluna}' não encontrada no DataFrame.")
        print(f"Erro ao criar histograma: Coluna '{coluna}' não encontrada.")
    except Exception as e:
        logging.error(f"Erro ao criar o histograma: {e}")
        print(f"Erro ao criar o histograma: {e}")

def criar_grafico_dispersao(df, coluna_x, coluna_y, output_dir):
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[coluna_x], df[coluna_y])
        plt.xlabel(coluna_x)
        plt.ylabel(coluna_y)
        plt.title(f'Gráfico de Dispersão de {coluna_x} vs {coluna_y}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'grafico_dispersao_{coluna_x}_vs_{coluna_y}.png'))
        logging.info(f"Gráfico de dispersão de {coluna_x} vs {coluna_y} criado e salvo com sucesso.")
        plt.show()
    except KeyError:
        logging.error(f"Coluna '{coluna_x}' ou '{coluna_y}' não encontrada no DataFrame.")
        print(f"Erro ao criar gráfico de dispersão: Coluna '{coluna_x}' ou '{coluna_y}' não encontrada.")
    except Exception as e:
        logging.error(f"Erro ao criar o gráfico de dispersão: {e}")
        print(f"Erro ao criar o gráfico de dispersão: {e}")

def verificar_e_criar_diretorio(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Diretório {output_dir} criado com sucesso.")
    else:
        logging.info(f"Diretório {output_dir} já existe.")

def escolher_colunas(df):
    while True:
        print("Colunas disponíveis:")
        print(df.columns.tolist())
        colunas = input("Digite os nomes das colunas que deseja analisar, separados por vírgula: ").split(',')
        colunas = [col.strip() for col in colunas]
        
        # Verificar se as colunas escolhidas são válidas
        colunas_invalidas = [col for col in colunas if col not in df.columns]
        if colunas_invalidas:
            print(f"Colunas inválidas: {colunas_invalidas}. Por favor, tente novamente.")
        else:
            return colunas

def escolher_graficos():
    print("Gráficos disponíveis:")
    print("1. Gráfico de Barras")
    print("2. Histograma")
    print("3. Gráfico de Dispersão")
    escolhas = input("Digite os números dos gráficos que deseja gerar, separados por vírgula: ").split(',')
    return [int(escolha.strip()) for escolha in escolhas]

def main(input_file, output_file, output_csv_file, json_file, sqlite_file, relatorio_file, output_dir):
    verificar_e_criar_diretorio(output_dir)
    
    df = ler_dados(input_file)
    df_cleaned = limpeza_dados(df)
    df_normalized = normalizar_dados(df_cleaned)
    df_filtrado = aplicar_filtros(df_normalized)
    df_transformado = transformar_dados(df_filtrado)
    estatisticas = analise_estatistica(df_transformado)

    print("Dados após limpeza:", df_cleaned.head(), sep="\n")
    print("Dados após normalização:", df_normalized.head(), sep="\n")
    print("Dados após filtros:", df_filtrado.head(), sep="\n")
    print("Dados após transformações:", df_transformado.head(), sep="\n")
    
    df_agrupado = df_transformado.groupby('County')['DOL Vehicle ID'].mean().reset_index()
    df_agrupado = df_agrupado.rename(columns={'DOL Vehicle ID': 'Media_DOL_Vehicle_ID'})
    df_agrupado['D'] = df_agrupado['Media_DOL_Vehicle_ID'] * 1.5

    print("Dados processados:")
    print(df_agrupado.head())

    if os.path.exists(output_file):
        resposta = input(f"O arquivo {output_file} já existe. Deseja sobrescrever? (s/n): ")
        if resposta.lower() != 's':
            logging.info(f"Exportação para Excel cancelada: {output_file}")
            print("Exportação para Excel cancelada.")
            return
    df_agrupado.to_excel(output_file, index=False)
    logging.info(f"Dados exportados para Excel em: {output_file}")

    if os.path.exists(output_csv_file):
        resposta = input(f"O arquivo {output_csv_file} já existe. Deseja sobrescrever? (s/n): ")
        if resposta.lower() != 's':
            logging.info(f"Exportação para CSV cancelada: {output_csv_file}")
            print("Exportação para CSV cancelada.")
            return
    df_agrupado.to_csv(output_csv_file, index=False)
    logging.info(f"Dados exportados para CSV em: {output_csv_file}")

    exportar_para_json(df_agrupado, json_file)
    exportar_para_sqlite(df_agrupado, sqlite_file)

    gerar_relatorio(df_agrupado, estatisticas, relatorio_file, output_dir)

    print("Processamento concluído. Dados salvos em:", output_file, output_csv_file, json_file, sqlite_file)

    # Escolher colunas para gráficos
    colunas = escolher_colunas(df_filtrado)

    # Escolher gráficos a serem gerados
    graficos = escolher_graficos()

    # Criar gráficos e salvar no diretório de saída
    if 1 in graficos:
        criar_grafico_barras(df_agrupado, output_dir)
    if 2 in graficos:
        for coluna in colunas:
            criar_histograma(df_filtrado, coluna, output_dir)
    if 3 in graficos and len(colunas) >= 2:
        criar_grafico_dispersao(df_filtrado, colunas[0], colunas[1], output_dir)

if __name__ == "__main__":
    print(f"Número de argumentos: {len(sys.argv)}")
    print(f"Argumentos: {sys.argv}")

    if len(sys.argv) < 8:
        print("Uso: python processar_planilha.py <input_file> <output_file> <output_csv_file> <json_file> <sqlite_file> <relatorio_file> <output_dir>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        output_csv_file = sys.argv[3]
        json_file = sys.argv[4]
        sqlite_file = sys.argv[5]
        relatorio_file = sys.argv[6]
        output_dir = sys.argv[7]
        main(input_file, output_file, output_csv_file, json_file, sqlite_file, relatorio_file, output_dir)

