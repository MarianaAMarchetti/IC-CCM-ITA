import cv2 
import os 
from datetime import datetime 
# Fotos: WebCam 
# Condição: Luz Natural
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
pasta_para_fotos = os.path.join(desktop, "Webcam_luz_natural")

if not os.path.exists(pasta_para_fotos):
    os.makedirs(pasta_para_fotos)

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    cv2.imshow("Pressione A para retirar a foto:", frame)

    key = cv2.waitKey(1)

    if key == ord('a'):
        data_e_hora_da_foto = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nome_arquivo_jpeg = f"Webcam_luz_natural_{data_e_hora_da_foto}.jpeg"
        diretorio = os.path.join(pasta_para_fotos, nome_arquivo_jpeg)

        cv2.imwrite(diretorio, frame)
        print(f"Foto salva em: {diretorio}")

    if key == ord('q'):
        break

webcam.release() 
cv2.destroyAllWindows()