# Análise da segmentação dos modelos de acordo com a área dos labels 
# Comparação entre as câmeras RealSense D400 e Webcam comum

# Por Mariana Ament Marchetti - Junho de 2026
# Iniciação Científica - Centro de Competência em Manufatura (CCM) - ITA

import os
import cv2
import csv 
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment 


# Funções de cálculo
def calcular_area_poligono(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calcular_centroide(vertices):
    momentos = cv2.moments(vertices.astype(np.int32))
    if momentos["m00"] == 0:
        return np.mean(vertices, axis=0)
    return np.array([int(momentos["m10"] / momentos["m00"]), int(momentos["m01"] / momentos["m00"])])

def carregar_labels_originais(txt_path, img_w, img_h):
    objetos = []
    if not os.path.exists(txt_path):
        return objetos
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        for linha in f.readlines():
            partes = linha.strip().split()
            if len(partes) < 5:
                continue  
                
            classe = int(partes[0])
            valores = [float(x) for x in partes[1:]]
            
            if len(partes) == 5: # Caixas
                cx_norm, cy_norm, w_norm, h_norm = valores
                cx, cy = cx_norm * img_w, cy_norm * img_h
                w_box, h_box = w_norm * img_w, h_norm * img_h
                x1, y1 = int(cx - w_box/2), int(cy - h_box/2)
                x2, y2 = int(cx + w_box/2), int(cy + h_box/2)
                coords = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                area = w_box * h_box
                centroide = np.array([int(cx), int(cy)])
            else: # Polígonos
                coords = np.array(valores).reshape(-1, 2)
                coords[:, 0] *= img_w
                coords[:, 1] *= img_h
                coords = coords.astype(np.int32)
                area = calcular_area_poligono(coords)
                centroide = calcular_centroide(coords)
            
            objetos.append({'classe': classe, 'poligono': coords, 'area': area, 'centroide': centroide})
    return objetos

# Função principal de processamento e coleta de dados dos modelos
def analise_total_de_todos_os_modelos():

    # Diretórios de saída dos dados coletados
    PATH_DESTINO_GERAL = "../OpenCV/Teste"
    FILE_CSV_UNICO = os.path.join(PATH_DESTINO_GERAL, "Analise_total_modelos.csv") 
     
    # Modelos a serem processados
    ANALISES = {
        "Modelo_Webcam_altura": {
            "PATH_IMAGES": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Datasets_webcam_final\\Altura\\test\\images",
            "PATH_LABELS": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Datasets_webcam_final\\Altura\\test\\labels",
            "PATH_MODEL": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_webcam_PRONTOS\\webcam_altura_YOLO26.pt",
            "MAPEAMENTO_CLASSES": {
                0: "Cilindro", 1: "Esfera", 2: "Paralelepipedo", 3: "Piramide", 4: "Prisma_hexagonal" 
            }
        },
        "Modelo_Webcam_iluminacao": {
            "PATH_IMAGES": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_webcam_treinados_final\\webcam_iluminacao\\dataset\\test\\images",
            "PATH_LABELS": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_webcam_treinados_final\\webcam_iluminacao\\dataset\\test\\labels",
            "PATH_MODEL": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_webcam_PRONTOS\\webcam_iluminacao_YOLO26.pt",
            "MAPEAMENTO_CLASSES": {
               0: "Cilindro", 1: "Esfera", 2: "Paralelepipedo", 3: "Piramide", 4: "Prisma_hexagonal" 
            }
        },
        "Modelo_Webcam_processamento": {
            "PATH_IMAGES": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_webcam_treinados_final\\webcam_completa\\dataset\\test\\images",
            "PATH_LABELS": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_webcam_treinados_final\\webcam_completa\\dataset\\test\\labels",
            "PATH_MODEL": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_webcam_PRONTOS\\webcam_process_YOLO26.pt",
            "MAPEAMENTO_CLASSES": {
               0: "Cilindro", 1: "Esfera", 2: "Paralelepipedo", 3: "Piramide", 4: "Prisma_hexagonal" 
            }
        },
        "Modelo_RealSense_altura": {
            "PATH_IMAGES": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_RealSense_treinados_final\\Datasets_RealSense_final\\Altura\\test\\images",
            "PATH_LABELS": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_RealSense_treinados_final\\Datasets_RealSense_final\\Altura\\test\\labels",
            "PATH_MODEL": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_RealSense_PRONTOS\\RealSense_Altura_YOLO26.pt",
            "MAPEAMENTO_CLASSES": {
               0: "Cilindro", 1: "Esfera", 2: "Paralelepipedo", 3: "Piramide", 4: "Prisma_hexagonal" 
            }
        },
        "Modelo_RealSense_iluminacao": {
            "PATH_IMAGES": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_RealSense_treinados_final\\Datasets_RealSense_final\\Iluminacao\\test\\images",
            "PATH_LABELS": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_RealSense_treinados_final\\Datasets_RealSense_final\\Iluminacao\\test\\labels",
            "PATH_MODEL": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_RealSense_PRONTOS\\RealSense_Iluminacao_YOLO26.pt",
            "MAPEAMENTO_CLASSES": {
               0: "Cilindro", 1: "Esfera", 2: "Paralelepipedo", 3: "Piramide", 4: "Prisma_hexagonal" 
            }
        },
        "Modelo_RealSense_processamento": {
            "PATH_IMAGES": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_RealSense_treinados_final\\Datasets_RealSense_final\\Completo\\test\\images",
            "PATH_LABELS": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_RealSense_treinados_final\\Datasets_RealSense_final\\Completo\\test\\labels",
            "PATH_MODEL": "C:\\Users\\mariana.marchetti\\Desktop\\Mariana\\OpenCV\\Modelos_RealSense_PRONTOS\\RealSense_process_YOLO26.pt",
            "MAPEAMENTO_CLASSES": {
               0: "Cilindro", 1: "Esfera", 2: "Paralelepipedo", 3: "Piramide", 4: "Prisma_hexagonal" 
            }
        }
    }

    # Estrutura: relatorios_modelos["Modelo_X"]["Tabela_Y"] = [linhas]
    relatorios_modelos = {}

    for nome_analise, caminhos in ANALISES.items():
        print(f"\n" + "="*50 + f"\nIniciando análise: {nome_analise}\n" + "="*50)
        
        PATH_IMAGES = caminhos["PATH_IMAGES"]
        PATH_LABELS = caminhos["PATH_LABELS"]
        MAPEAMENTO_CLASSES = caminhos["MAPEAMENTO_CLASSES"]
        
        PATH_OUTPUT_MODELO = os.path.join(PATH_DESTINO_GERAL, nome_analise)
        os.makedirs(PATH_OUTPUT_MODELO, exist_ok=True)

        print("Carregando modelo.")
        try:
            model = YOLO(caminhos["PATH_MODEL"])
            print("Modelo carregado.")
        except Exception as e:
            print(f"Falha ao carregar modelo {nome_analise}: {e}")
            continue 

        formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp')
        lista_imagens = [f for f in os.listdir(PATH_IMAGES) if f.lower().endswith(formatos_validos)]
        total_imagens = len(lista_imagens)
        
        print(f"{total_imagens} imagens encontradas. Segmentação em processamento (conf=0.8).")

        caminhos_completos = [os.path.join(PATH_IMAGES, f) for f in lista_imagens]
        resultados_yolo = model(caminhos_completos, conf=0.8, verbose=False, stream=True)

        dados_tabela_detalhada = [] 
        lista_erros_sem_par = [] 
        dados_erros_por_imagem = [] 
        
        erros_apenas_novos = 0
        erros_nao_detectados = 0
        acumulador_area_descartada = 0.0

        metricas_por_classe = {nome: [0.0, 0.0, 0.0, 0] for nome in MAPEAMENTO_CLASSES.values()}
        classe_desconhecida_chave = "Outros/Não Mapeados"
        metricas_por_classe[classe_desconhecida_chave] = [0.0, 0.0, 0.0, 0]

        acumulador_area_original = 0
        acumulador_area_nova = 0

        print(f"Processando segmentações e salvando imagens em: '{PATH_OUTPUT_MODELO}'")
        
        for i, (nome_img, resultado) in enumerate(zip(lista_imagens, resultados_yolo), start=1):
            caminho_img = os.path.join(PATH_IMAGES, nome_img)
            nome_base = os.path.splitext(nome_img)[0]
            caminho_txt = os.path.join(PATH_LABELS, f"{nome_base}.txt")

            img = cv2.imread(caminho_img)
            if img is None:
                continue
            h, w, _ = img.shape

            labels_originais = carregar_labels_originais(caminho_txt, w, h)

            labels_novos = []
            if resultado.masks is not None:
                poligonos_novos = resultado.masks.xy
                classes_novas = resultado.boxes.cls.cpu().numpy()

                for classe, poli in zip(classes_novas, poligonos_novos):
                    if len(poli) > 0:
                        labels_novos.append({
                            'classe': int(classe), 
                            'poligono': poli.astype(np.int32), 
                            'area': calcular_area_poligono(poli),
                            'centroide': calcular_centroide(poli)
                        })

            pares_encontrados = []
            originais_pareados = set()
            novos_pareados = set()

            DISTANCIA_MAXIMA_PERMITIDA = 30 

            n_orig = len(labels_originais)
            n_novos = len(labels_novos)

            if n_orig > 0 and n_novos > 0:
                matriz_custo = np.full((n_orig, n_novos), fill_value=1e6)

                for idx_orig, obj_orig in enumerate(labels_originais):
                    for idx_novo, obj_novo in enumerate(labels_novos):
                        if obj_orig['classe'] == obj_novo['classe']:
                            dist = np.linalg.norm(obj_orig['centroide'] - obj_novo['centroide'])
                            matriz_custo[idx_orig, idx_novo] = dist

                linha_ind, coluna_ind = linear_sum_assignment(matriz_custo)

                for r, c in zip(linha_ind, coluna_ind):
                    if matriz_custo[r, c] <= DISTANCIA_MAXIMA_PERMITIDA:
                        pares_encontrados.append((r, c))
                        originais_pareados.add(r)
                        novos_pareados.add(c)
                    else:
                        nome_cl = MAPEAMENTO_CLASSES.get(labels_originais[r]['classe'], "Desconhecida")
                        print(f"[CORREÇÃO] Par rejeitado na imagem {nome_img}: Classe {nome_cl} remota ({matriz_custo[r, c]:.1f}px)")

            for idx_orig in range(n_orig):
                if idx_orig not in originais_pareados:
                    pares_encontrados.append((idx_orig, None))

            for idx_novo in range(n_novos):
                if idx_novo not in novos_pareados:
                    pares_encontrados.append((None, idx_novo))

            contagem_pares_validos_imagem = 0
            contagem_erros_imagem = 0

            id_objeto_relatorio = 1
            for idx_orig, idx_novo in pares_encontrados:
                
                if idx_orig is None or idx_novo is None:
                    contagem_erros_imagem += 1
                    if idx_orig is None:
                        obj_err = labels_novos[idx_novo]
                        classe_err = obj_err['classe']
                        nome_cl_err = MAPEAMENTO_CLASSES.get(classe_err, f"Classe {classe_err}")
                        lista_erros_sem_par.append([nome_img, f"Apenas Novo ({nome_cl_err})", f"{obj_err['area']:.2f} px²"])
                        erros_apenas_novos += 1
                        acumulador_area_descartada += obj_err['area']
                        
                        cv2.polylines(img, [obj_err['poligono']], isClosed=True, color=(0, 0, 255), thickness=1)
                        cv2.circle(img, tuple(obj_err['centroide']), 2, (0, 0, 255), -1)
                    else:
                        obj_err = labels_originais[idx_orig]
                        classe_err = obj_err['classe']
                        nome_cl_err = MAPEAMENTO_CLASSES.get(classe_err, f"Classe {classe_err}")
                        lista_erros_sem_par.append([nome_img, f"Não Detectado ({nome_cl_err})", f"{obj_err['area']:.2f} px²"])
                        erros_nao_detectados += 1
                        acumulador_area_descartada += obj_err['area']
                        
                        cv2.polylines(img, [obj_err['poligono']], isClosed=True, color=(0, 255, 0), thickness=1)
                        cv2.circle(img, tuple(obj_err['centroide']), 2, (0, 255, 0), -1)
                    
                    continue 
                
                contagem_pares_validos_imagem += 1
                obj_orig = labels_originais[idx_orig]
                obj_novo = labels_novos[idx_novo]
                
                area_orig = obj_orig['area']
                area_nova = obj_novo['area']
                classe_int = obj_orig['classe']
                
                ponto_orig_centroide = tuple(obj_orig['centroide'])
                ponto_novo_centroide = tuple(obj_novo['centroide'])

                cv2.polylines(img, [obj_orig['poligono']], isClosed=True, color=(0, 255, 0), thickness=1)
                cv2.circle(img, ponto_orig_centroide, 2, (0, 255, 0), -1)
                cv2.polylines(img, [obj_novo['poligono']], isClosed=True, color=(0, 0, 255), thickness=1)
                cv2.circle(img, ponto_novo_centroide, 2, (0, 0, 255), -1)
                cv2.line(img, ponto_orig_centroide, ponto_novo_centroide, (0, 255, 255), 1)

                nome_classe_customizado = MAPEAMENTO_CLASSES.get(classe_int, f"Classe {classe_int}")
                if nome_classe_customizado not in metricas_por_classe:
                    metricas_por_classe[nome_classe_customizado] = [0.0, 0.0, 0.0, 0]

                diferenca_objeto = area_nova - area_orig
                porcentagem_objeto = (diferenca_objeto / area_orig) * 100
                porcentagem_str = f"{porcentagem_objeto:+.2f}%"

                acumulador_area_original += area_orig
                acumulador_area_nova += area_nova

                status_txt = f"Objeto {id_objeto_relatorio} ({nome_classe_customizado})"

                erro_absolute = abs(diferenca_objeto)
                porcentagem_erro = (erro_absolute / area_orig) * 100
                porcentagem_erro_str = f"{porcentagem_erro:.2f}%"

                dados_tabela_detalhada.append([
                    nome_img, status_txt, f"{area_orig:.2f} px²", f"{area_nova:.2f} px²", 
                    f"{diferenca_objeto:+.2f} px²", f"{erro_absolute:.2f} px²", porcentagem_str, porcentagem_erro_str, porcentagem_erro       
                ])

                metricas_por_classe[nome_classe_customizado][0] += area_orig
                metricas_por_classe[nome_classe_customizado][1] += area_nova
                metricas_por_classe[nome_classe_customizado][2] += porcentagem_erro
                metricas_por_classe[nome_classe_customizado][3] += 1

                id_objeto_relatorio += 1

            total_entidades_imagem = contagem_pares_validos_imagem + contagem_erros_imagem
            taxa_erro_imagem = (contagem_erros_imagem / total_entidades_imagem) * 100 if total_entidades_imagem > 0 else 0.0
            dados_erros_por_imagem.append([nome_img, contagem_pares_validos_imagem, contagem_erros_imagem, f"{taxa_erro_imagem:.2f}%", taxa_erro_imagem])

            caminho_saida_img = os.path.join(PATH_OUTPUT_MODELO, f"processada_{nome_img}")
            cv2.imwrite(caminho_saida_img, img)

            if i % 10 == 0 or i == total_imagens:
                print(f"    [{i}/{total_imagens}] imagens processadas...")

        # Cálculos das métricas principais
        diferenca_total_absoluta = acumulador_area_nova - acumulador_area_original
        porcentagem_total_relativa = (diferenca_total_absoluta / acumulador_area_original) * 100 if acumulador_area_original > 0 else 0.0

        area_total_geral_computada = acumulador_area_original + acumulador_area_nova + acumulador_area_descartada
        porcentagem_area_descartada = (acumulador_area_descartada / area_total_geral_computada) * 100 if area_total_geral_computada > 0 else 0.0

        erros_float = [linha[-1] for linha in dados_tabela_detalhada]
        erro_medio_porcentagem = sum(erros_float) / len(erros_float) if erros_float else 0.0
        taxa_erro_medio_global_detec = sum([row[-1] for row in dados_erros_por_imagem]) / len(dados_erros_por_imagem) if dados_erros_por_imagem else 0.0

        # Separação das tabelas por chaves estruturadas para o alinhamento 
        tabelas_modelo = {}
        
        # Tabela 1: Métricas Principais
        t1 = []
        t1.append([f"Tabela 1: Métricas principais do dataset ({nome_analise})", ""])
        t1.append(["Métrica", "Valor"])
        t1.append(["Total das áreas originais", f"{acumulador_area_original:.2f} px²"])
        t1.append(["Total das áreas novas", f"{acumulador_area_nova:.2f} px²"])
        t1.append(["Diferença absoluta nas áreas pareadas", f"{diferenca_total_absoluta:+.2f} px²"]) 
        t1.append(["Porcentagem relativa da variação total", f"{porcentagem_total_relativa:+.2f}%"]) 
        t1.append(["Erro médio por entidade pareada (%)", f"{erro_medio_porcentagem:.2f}%"]) 
        t1.append(["Soma total de área descartada", f"{acumulador_area_descartada:.2f} px²"]) 
        t1.append(["Percentual de área descartada", f"{porcentagem_area_descartada:.2f}%"]) 
        t1.append(["Módulo (%) de área descartada", f"{abs(porcentagem_area_descartada):.2f}%"])
        t1.append(["Média global do erro de detecção por imagem", f"{taxa_erro_medio_global_detec:.2f}%"])
        t1.append(["Total de entidades pareadas com sucesso", len(dados_tabela_detalhada)]) 
        t1.append(["Total de erros de detecção", len(lista_erros_sem_par)]) 
        tabelas_modelo["Tabela1"] = t1

        # Tabela 2: Análise por classe
        t2 = []
        t2.append(["Tabela 2: Análise acumulada por classe de objeto", "", "", "", "", ""])
        t2.append(["Classe do objeto", "Área Original", "Área Nova", "Diferença Absoluta", "Erro médio da classe", "Objetos processados"])
        for nome_classe, valores in metricas_por_classe.items():
            soma_orig, soma_nova, soma_erros_pct, qtd = valores
            if qtd == 0:
                continue
            dif_acumulada = soma_nova - soma_orig
            erro_medio_classe = soma_erros_pct / qtd
            t2.append([nome_classe, f"{soma_orig:.2f} px²", f"{soma_nova:.2f} px²", f"{dif_acumulada:+.2f} px²", f"{erro_medio_classe:.2f}%", qtd])
        tabelas_modelo["Tabela2"] = t2

        # Tabela 3: Detalhamento individual
        t3 = []
        t3.append(["Tabela 3: Detalhamento individual por objeto pareado", "", "", "", "", "", "", ""])
        t3.append(["Imagem", "Identificação", "Área original (Verde)", "Área nova (Vermelha)", "Diferença (Vermelha-Verde)", "Erro absoluto (px²)", "Variação (%)", "Módulo do erro (%)"])
        for linha in dados_tabela_detalhada:
            t3.append(linha[:-1]) 
        tabelas_modelo["Tabela3"] = t3

        # Tabela 4: Métricas por imagem
        t4 = []
        t4.append(["Tabela 4: Métrica de erros de detecção individual por imagem", "", "", ""])
        t4.append(["Imagem", "Objetos pareados (sucesso)", "Objetos sem par (erros)", "Taxa de erro da imagem (%)"])
        for row in dados_erros_por_imagem:
            t4.append(row[:-1])
        tabelas_modelo["Tabela4"] = t4

        # Tabela 5: Contabilidade de erros
        t5 = []
        t5.append(["Tabela 5: Objetos sem par correspondente", f"Apenas novos: {erros_apenas_novos}", f"Não detectados: {erros_nao_detectados}"])
        t5.append(["Imagem", "Tipo do erro", "Área descartada"])
        for err in lista_erros_sem_par:
            t5.append(err)
        tabelas_modelo["Tabela5"] = t5
            
        relatorios_modelos[nome_analise] = tabelas_modelo
        print(f"Análise de {nome_analise} concluída com sucesso.")


    # Organização estruturada das tabelas no arquivo .csv
    print("\nGerando arquivo .csv estruturado...")
    try:
        os.makedirs(PATH_DESTINO_GERAL, exist_ok=True)
        
        modelos_ordenados = list(ANALISES.keys())
        LARGURA_MODELO = 8 
        ESPACAMENTO_V = 3  # Intervalos de 3 linhas em branco entre as tabelas
        
        # Dicionário para salvar todas as células mapeadas por (linha, coluna)
        grade_csv = {}
        
        # Controla o tamanho máximo que a grade vai atingir verticalmente
        max_linha_global = 0

        # Constrói a grade na memória iterando modelo por modelo (horizontalmente independente)
        for idx_mod, mod in enumerate(modelos_ordenados):
            col_inicial = idx_mod * (LARGURA_MODELO + 1)
            linha_atual = 0  # Reseta o cursor vertical para o topo a cada modelo
            
            for id_tabela in ["Tabela1", "Tabela2", "Tabela3", "Tabela4", "Tabela5"]:
                if mod in relatorios_modelos and id_tabela in relatorios_modelos[mod]:
                    dados_tabela = relatorios_modelos[mod][id_tabela]
                    
                    # Escreve a tabela linha por linha nas coordenadas exclusivas desse modelo
                    for linha_tab in dados_tabela:
                        for idx_col, valor in enumerate(linha_tab):
                            if idx_col < LARGURA_MODELO:
                                grade_csv[(linha_atual, col_inicial + idx_col)] = valor
                        linha_atual += 1
                    
                    # Após fechar essa tabela desse modelo, tem intervalo de 3 linhas vazias
                    linha_atual += ESPACAMENTO_V
            
            # Registra qual foi a maior linha alcançada no total geral do documento
            if linha_atual > max_linha_global:
                max_linha_global = linha_atual

        # Calcula o número total de colunas necessárias para acomodar os 6 modelos lado a lado
        max_coluna_global = len(modelos_ordenados) * (LARGURA_MODELO + 1) - 1

        # Grava os dados da grade mapeada no arquivo CSV físico de cima para baixo
        with open(FILE_CSV_UNICO, mode='w', newline='', encoding='utf-8-sig') as f_csv:
            writer = csv.writer(f_csv, delimiter=';')
            
            for r in range(max_linha_global):
                linha_dados = []
                linha_vazia = True
                
                for c in range(max_coluna_global):
                    celula = grade_csv.get((r, c), "")
                    if celula != "":
                        linha_vazia = False
                    linha_dados.append(celula)
                
                # Se a linha global do arquivo for 100% vazia, escreve uma linha de quebra pura.
                if linha_vazia:
                    writer.writerow([])
                else:
                    writer.writerow(linha_dados)

        print(f"\nO arquivo com todas as tabelas de cada modelo foi criado com sucesso em: '{FILE_CSV_UNICO}'")
    except Exception as e:
        print(f"Falha ao gravar arquivo .csv: {e}")


if __name__ == "__main__":
    analise_total_de_todos_os_modelos()