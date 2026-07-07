# Análise 01 dos modelos (tempo de detecção, média e desvio padrão do limiar de confiança)
# Análise completa das duas câmeras (WebCam e RealSense)

# Por Mariana Ament Marchetti - Junho de 2026
# Iniciação Científica - Centro de Competência em Manufatura (CCM) - ITA

import os
import time 
import numpy as np
from ultralytics import YOLO 
from tabulate import tabulate

def analisar_altura_webcam(modelo_path, pasta_imagens):
    
    # Carregamento do modelo variando altura
    print("\n")
    print(f"Carregando o modelo para análise de altura: {modelo_path}...\n")
    model = YOLO(modelo_path)   
    formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    imagens = [os.path.join(pasta_imagens, f) for f in os.listdir(pasta_imagens) if f.lower().endswith(formatos_validos)] 

    if not imagens:
        print(f"Nenhuma imagem válida encontrada em: {pasta_imagens}")
        return

    todas_confiancas = []
    print(f"Iniciando a inferência em {len(imagens)} imagens variando altura...")

    for img_path in imagens:
        # Processamento (segmentação) das imagens
        results = model(img_path, verbose=False)
        
        for result in results:
            if result.boxes is not None and result.boxes.conf is not None:
                conf_list = result.boxes.conf.cpu().numpy().tolist()
                todas_confiancas.extend(conf_list)

    # Cálculos estatísticos
    if todas_confiancas:
        media_conf = np.mean(todas_confiancas)     
        desvio_padrao_conf = np.std(todas_confiancas)
    else:
        media_conf, desvio_padrao_conf = 0.0, 0.0

    # Configurações da tabela dos dados de altura
    tabela_confianca = [
        ["Média (µ)", f"{media_conf:.4f}", f"{media_conf * 100:.2f}%"],
        ["Desvio Padrão (σ)", f"{desvio_padrao_conf:.4f}", f"{desvio_padrao_conf * 100:.2f}%"]
    ]
    headers_confianca = ["Métrica de Confiança", "Valor (0 a 1)", "Porcentagem (%)"]

    print("\n" + "-"*33)
    print("Análise variando altura (WebCam)")
    print("-"*33)
    print(tabulate(tabela_confianca, headers=headers_confianca, tablefmt="fancy_grid"))
    print("\n")

def analisar_iluminacao_webcam(modelo_path, pasta_imagens):

    # Carregamento do modelo variando iluminação 
    print(f"Carregando o modelo para análise de iluminação: {modelo_path}...\n")
    model = YOLO(modelo_path)
    formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    imagens = [os.path.join(pasta_imagens, f) for f in os.listdir(pasta_imagens) if f.lower().endswith(formatos_validos)] 

    if not imagens:
        print(f"Nenhuma imagem válida encontrada em: {pasta_imagens}")
        return

    todas_confiancas = []
    print(f"Iniciando a inferência em {len(imagens)} imagens variando iluminação...")
    
    for img_path in imagens:
        # Processamento (segmentação) das imagens 
        results = model(img_path, verbose=False)
        
        for result in results:
            if result.boxes is not None and result.boxes.conf is not None:
                conf_list = result.boxes.conf.cpu().numpy().tolist()
                todas_confiancas.extend(conf_list)

    # Cálculos estatísticos
    if todas_confiancas:
        media_conf = np.mean(todas_confiancas) 
        desvio_padrao_conf = np.std(todas_confiancas)
    else:
        media_conf, desvio_padrao_conf = 0.0, 0.0

    # Configurações da tabela dos dados de iluminação
    tabela_confianca = [
        ["Média (µ)", f"{media_conf:.4f}", f"{media_conf * 100:.2f}%"],
        ["Desvio Padrão (σ)", f"{desvio_padrao_conf:.4f}", f"{desvio_padrao_conf * 100:.2f}%"]
    ]
    headers_confianca = ["Métrica de Confiança", "Valor (0 a 1)", "Porcentagem (%)"]

    print("\n" + "-"*37)
    print("Análise variando iluminação (WebCam)")
    print("-"*37)
    print(tabulate(tabela_confianca, headers=headers_confianca, tablefmt="fancy_grid"))
    print("\n")

def analisar_processamento_webcam(modelo_path, pasta_imagens):
    
    # Carregamento do modelo para análise de processamento
    print(f"Carregando o modelo: {modelo_path}...\n")
    model = YOLO(modelo_path)
    formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    imagens = [os.path.join(pasta_imagens, f) for f in os.listdir(pasta_imagens) if f.lower().endswith(formatos_validos)]

    if not imagens:
        print(f"Nenhuma imagem válida encontrada em: {pasta_imagens}")
        return

    todas_confiancas = []
    tempos_processamento = []

    print(f"Iniciando processamento de {len(imagens)} imagens...")

    # Processamento das imagens (com a medição do tempo)
    for img_path in imagens:
        start_time = time.time()
        results = model(img_path, verbose=False)
        end_time = time.time()
        
        tempo_inferencia = (end_time - start_time) * 1000  # Tempo em milissegundos
        tempos_processamento.append(tempo_inferencia)
        
        for result in results:
            if result.boxes is not None and result.boxes.conf is not None:
                conf_list = result.boxes.conf.cpu().numpy().tolist()
                todas_confiancas.extend(conf_list)

    # Cálculos para dados referentes ao tempo
    media_tempo = np.mean(tempos_processamento)
    tempo_total = np.sum(tempos_processamento) / 1000  

    # Configurações da tabela para exibição dos dados
    tabela_tempo = [
        ["Imagens Processadas", f"{len(imagens)}"],
        ["Objetos Segmentados", f"{len(todas_confiancas)}"],
        ["Tempo Médio / Imagem", f"{media_tempo:.2f} ms"],
        ["Tempo Total de Execução", f"{tempo_total:.2f} s"]
    ]
    headers_tempo = ["Parâmetro de performance", "Resultado"]

    print("\n" + "-"*37)
    print("Análise de processamento (WebCam)")
    print("-"*37)
    print(tabulate(tabela_tempo, headers=headers_tempo, tablefmt="fancy_grid"))
    print("\n")

def analisar_altura_RealSense(modelo_path, pasta_imagens):
    
    # Carregamento do modelo variando altura
    print("\n")
    print(f"Carregando o modelo para análise de altura: {modelo_path}...\n")
    model = YOLO(modelo_path)   
    formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    imagens = [os.path.join(pasta_imagens, f) for f in os.listdir(pasta_imagens) if f.lower().endswith(formatos_validos)] 

    if not imagens:
        print(f"Nenhuma imagem válida encontrada em: {pasta_imagens}")
        return

    todas_confiancas = []
    print(f"Iniciando a inferência em {len(imagens)} imagens variando altura...")

    for img_path in imagens:
        # Processamento (segmentação) das imagens
        results = model(img_path, verbose=False)
        
        for result in results:
            if result.boxes is not None and result.boxes.conf is not None:
                conf_list = result.boxes.conf.cpu().numpy().tolist()
                todas_confiancas.extend(conf_list)

    # Cálculos estatísticos
    if todas_confiancas:
        media_conf = np.mean(todas_confiancas)     
        desvio_padrao_conf = np.std(todas_confiancas)
    else:
        media_conf, desvio_padrao_conf = 0.0, 0.0

    # Configurações da tabela dos dados de altura
    tabela_confianca = [
        ["Média (µ)", f"{media_conf:.4f}", f"{media_conf * 100:.2f}%"],
        ["Desvio Padrão (σ)", f"{desvio_padrao_conf:.4f}", f"{desvio_padrao_conf * 100:.2f}%"]
    ]
    headers_confianca = ["Métrica de Confiança", "Valor (0 a 1)", "Porcentagem (%)"]

    print("\n" + "-"*33)
    print("Análise variando altura (RealSense)")
    print("-"*33)
    print(tabulate(tabela_confianca, headers=headers_confianca, tablefmt="fancy_grid"))
    print("\n")

def analisar_iluminacao_RealSense(modelo_path, pasta_imagens):

    # Carregamento do modelo variando iluminação 
    print(f"Carregando o modelo para análise de iluminação: {modelo_path}...\n")
    model = YOLO(modelo_path)
    formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    imagens = [os.path.join(pasta_imagens, f) for f in os.listdir(pasta_imagens) if f.lower().endswith(formatos_validos)] 

    if not imagens:
        print(f"Nenhuma imagem válida encontrada em: {pasta_imagens}")
        return

    todas_confiancas = []
    print(f"Iniciando a inferência em {len(imagens)} imagens variando iluminação...")
    
    for img_path in imagens:
        # Processamento (segmentação) das imagens 
        results = model(img_path, verbose=False)
        
        for result in results:
            if result.boxes is not None and result.boxes.conf is not None:
                conf_list = result.boxes.conf.cpu().numpy().tolist()
                todas_confiancas.extend(conf_list)

    # Cálculos estatísticos
    if todas_confiancas:
        media_conf = np.mean(todas_confiancas) 
        desvio_padrao_conf = np.std(todas_confiancas)
    else:
        media_conf, desvio_padrao_conf = 0.0, 0.0

    # Configurações da tabela dos dados de iluminação
    tabela_confianca = [
        ["Média (µ)", f"{media_conf:.4f}", f"{media_conf * 100:.2f}%"],
        ["Desvio Padrão (σ)", f"{desvio_padrao_conf:.4f}", f"{desvio_padrao_conf * 100:.2f}%"]
    ]
    headers_confianca = ["Métrica de Confiança", "Valor (0 a 1)", "Porcentagem (%)"]

    print("\n" + "-"*37)
    print("Análise variando iluminação (RealSense)")
    print("-"*37)
    print(tabulate(tabela_confianca, headers=headers_confianca, tablefmt="fancy_grid"))
    print("\n")

def analisar_processamento_RealSense(modelo_path, pasta_imagens):
    
    # Carregamento do modelo para análise de processamento
    print(f"Carregando o modelo: {modelo_path}...\n")
    model = YOLO(modelo_path)
    formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    imagens = [os.path.join(pasta_imagens, f) for f in os.listdir(pasta_imagens) if f.lower().endswith(formatos_validos)]

    if not imagens:
        print(f"Nenhuma imagem válida encontrada em: {pasta_imagens}")
        return

    todas_confiancas = []
    tempos_processamento = []

    print(f"Iniciando processamento de {len(imagens)} imagens...")

    # Processamento das imagens (com a medição do tempo)
    for img_path in imagens:
        start_time = time.time()
        results = model(img_path, verbose=False)
        end_time = time.time()
        
        tempo_inferencia = (end_time - start_time) * 1000  # Tempo em milissegundos
        tempos_processamento.append(tempo_inferencia)
        
        for result in results:
            if result.boxes is not None and result.boxes.conf is not None:
                conf_list = result.boxes.conf.cpu().numpy().tolist()
                todas_confiancas.extend(conf_list)

    # Cálculos para dados referentes ao tempo
    media_tempo = np.mean(tempos_processamento)
    tempo_total = np.sum(tempos_processamento) / 1000  

    # Configurações da tabela para exibição dos dados
    tabela_tempo = [
        ["Imagens Processadas", f"{len(imagens)}"],
        ["Objetos Segmentados", f"{len(todas_confiancas)}"],
        ["Tempo Médio / Imagem", f"{media_tempo:.2f} ms"],
        ["Tempo Total de Execução", f"{tempo_total:.2f} s"]
    ]
    headers_tempo = ["Parâmetro de performance", "Resultado"]

    print("\n" + "-"*37)
    print("Análise de processamento (RealSense)")
    print("-"*37)
    print(tabulate(tabela_tempo, headers=headers_tempo, tablefmt="fancy_grid"))
    print("\n")


if __name__ == "__main__":
    # Função para análise de altura da WebCam
    CAMINHO_MODELO_ALTURA = "../Modelos_webcam_PRONTOS/webcam_altura_YOLO26.pt" 
    PASTA_DE_IMAGENS_ALTURA = "../Modelos_webcam_treinados_final/webcam_altura/dataset/test/images"
    
    analisar_altura_webcam(CAMINHO_MODELO_ALTURA, PASTA_DE_IMAGENS_ALTURA)

    # Função para análise de iluminação da WebCam
    CAMINHO_MODELO_ILUMINACAO = "../Modelos_webcam_PRONTOS/webcam_iluminacao_YOLO26.pt" 
    PASTA_DE_IMAGENS_ILUMINACAO = "../Modelos_webcam_treinados_final/webcam_iluminacao/dataset/test/images"
    
    analisar_iluminacao_webcam(CAMINHO_MODELO_ILUMINACAO, PASTA_DE_IMAGENS_ILUMINACAO)

    # Função para análise de processamento da WebCam
    CAMINHO_MODELO_PROCESSAMENTO = "../Modelos_webcam_PRONTOS/webcam_process_YOLO26.pt" 
    PASTA_DE_IMAGENS_PROCESSAMENTO = "../Modelos_webcam_treinados_final/webcam_completa/dataset/test/images"
    
    analisar_processamento_webcam(CAMINHO_MODELO_PROCESSAMENTO, PASTA_DE_IMAGENS_PROCESSAMENTO)

    # Função para análise de altura da RealSense
    CAMINHO_MODELO_ALTURA_RS = "../Modelos_RealSense_PRONTOS/RealSense_Altura_YOLO26.pt" 
    PASTA_DE_IMAGENS_ALTURA_RS = "../Modelos_RealSense_treinados_final/Datasets_RealSense_final/Altura/test/images"
    
    analisar_altura_RealSense(CAMINHO_MODELO_ALTURA_RS, PASTA_DE_IMAGENS_ALTURA_RS)

    # Função para análise de iluminação da RealSense
    CAMINHO_MODELO_ILUMINACAO_RS = "../Modelos_RealSense_PRONTOS/RealSense_Iluminacao_YOLO26.pt" 
    PASTA_DE_IMAGENS_ILUMINACAO_RS = "../Modelos_RealSense_treinados_final/Datasets_RealSense_final/Iluminacao/test/images"
    
    analisar_iluminacao_RealSense(CAMINHO_MODELO_ILUMINACAO_RS, PASTA_DE_IMAGENS_ILUMINACAO_RS)

    # Função para análise de processamento da RealSense
    CAMINHO_MODELO_PROCESSAMENTO_RS = "../Modelos_RealSense_PRONTOS/RealSense_process_YOLO26.pt" 
    PASTA_DE_IMAGENS_PROCESSAMENTO_RS = "../Modelos_RealSense_treinados_final/Datasets_RealSense_final/Completo/test/images"
    
    analisar_processamento_RealSense(CAMINHO_MODELO_PROCESSAMENTO_RS, PASTA_DE_IMAGENS_PROCESSAMENTO_RS)