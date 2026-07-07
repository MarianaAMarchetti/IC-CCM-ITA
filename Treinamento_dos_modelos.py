# Múltiplo treinamento de modelos YOLO26
# Junção dos dados de altura e iluminação para o treinamento do modelo com ambas condições

# Por Mariana Ament Marchetti - Junho de 2026
# Iniciação Científica - Centro de Competência em Manufatura (CCM) - ITA

import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

# Caminhos para os datasets
sources = [
    Path("Datasets_RealSense_final/Altura"),
    Path("Datasets_RealSense_final/Iluminacao"),
]
target = Path("Datasets_RealSense_final/Completo")

#Junção dos dois datasets em apenas 1
for split in ["train", "valid", "test"]:
    for kind in ["images", "labels"]:
        dest = target / split / kind
        dest.mkdir(parents=True, exist_ok=True)
        for src in sources:
            for file in (src / split / kind).glob("*"):
                dst = dest / file.name
                if dst.exists():
                    dst = dest.with_name(f"{src.parent.name}_{file.name}")
                shutil.copy(file, dst)


# Execução dos treinamentos dos modelos
if __name__ == "__main__":
    model_altura = YOLO("../model/yolo26s-seg.pt")
    model_iluminacao = YOLO("../model/yolo26s-seg.pt")
    model_todos = YOLO("../model/yolo26s-seg.pt")

    # Treinamento com dataset "Altura"
    altura = model_altura.train(
        data="Datasets_RealSense_final/Altura/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        exist_ok=True,
        save_dir="Treinado_Realsense_Altura" 
    )
    print(altura)

    # Treinamento com dataset "Iluminacao"
    iluminacao = model_iluminacao.train(
        data="Datasets_RealSense_final/Iluminacao/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        exist_ok=True,
        save_dir="Treinado_Realsense_Iluminacao"
    )
    print(iluminacao)

    # Treinamento com todos os datasets 
    todos = model_todos.train(
        data="Datasets_RealSense_final/Completo/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        exist_ok=True,
        save_dir="Treinado_Realsense_Todos"
    )
    print(todos)