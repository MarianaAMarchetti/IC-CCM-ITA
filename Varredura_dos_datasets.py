# Programa para varredura dos datasets para treinamento
# Retirada dos arquivos vazios presentes nas pastas

# Por Mariana Ament Marchetti - Junho de 2026
# Iniciação Científica - Centro de Competência em Manufatura (CCM) - ITA

import os

caminho_da_pasta = "c:\\Users\\mariana.marchetti\\Desktop\\Mariana\\Datasets_RealSense_MODIFICANDO\\Altura\\train\\labels"
 
def listar_e_remover_txt_vazios(caminho_da_pasta: str):

    remover = [] 

    if not os.path.isdir(caminho_da_pasta):
        print(f"A pasta '{caminho_da_pasta}' não existe.")
        return

    try:
       
        for item in os.listdir(caminho_da_pasta):
            caminho_item = os.path.join(caminho_da_pasta, item)
       
            if os.path.isfile(caminho_item) and item.lower().endswith(".txt"):
                if os.path.getsize(caminho_item) == 0:
                    remover.append(caminho_item)

        if remover:
            print("Arquivos .txt vazios encontrados:")
            for arquivo in remover:
                print(f" - {arquivo}")
                os.remove(arquivo)
                img = arquivo.replace("labels", "images")
                os.remove(img.replace("txt", "jpg"))
 
        else:
            print("Nenhum arquivo .txt vazio encontrado.")

    except Exception as e:
        print(f"Erro: {e}")

listar_e_remover_txt_vazios(caminho_da_pasta)