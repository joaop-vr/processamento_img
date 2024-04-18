import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

rois = []


"""# Variáveis delimitadoras da região de interesse (ROI)
x = 850
y = 660
w = 1560
h = 1110

# Recortar região de interesse (ROI)
roi = img[y:y+h, x:x+w]

# Salvar o ROI
cv2.imwrite("roi.png", roi)

# Opcional
cv2.rectangle(img, (x,y), (x+w, y+h), (0,255, 0), 2)



# Variáveis delimitadoras da região de interesse (ROI)
x = 850
y = 1770
w = 570
h = 200
# Opcional
cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 225), 2)

# Variáveis delimitadoras da região de interesse (ROI)
x = 70
y = 2070
w = 600
h = 250
# Opcional
cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 0), 2)

# Variáveis delimitadoras da região de interesse (ROI)
x = 1330
y = 2070
w = 700
h = 220
# Opcional
cv2.rectangle(img, (x,y), (x+w, y+h), (0,255, 255), 2)

# Variáveis delimitadoras da região de interesse (ROI)
x = 900
y = 2290
w = 1480
h = 220
# Opcional
cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)"""


def segment_form(img_path):

    # Carregar imagem
    img = cv2.imread(img_path)

    for i in range(len(rois)):

        # Variáveis delimitadoras da região de interesse (ROI)
        x, y, w, h = rois[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)

        # Recortar região de interesse (ROI)
        roi = img[y:y+h, x:x+w]

        # Salvar o ROI
        filename = "roi"+str(i) +".png"
        cv2.imwrite(filename, roi)


        img_segmented = cv2.imread(filename)
        # Calcular o histograma ao longo do eixo X
        histogram = np.sum(img_segmented, axis=0)

        # Normalizar o histograma
        histogram = histogram / np.max(histogram)

        # Plotar
        plt.figure(figsize=(10, 5))
        plt.plot(histogram, color='black')
        title = 'Projeção do Histograma' + str(i) + 'ao Longo do Eixo X'
        plt.title(title)
        plt.xlabel('Posição no Eixo X')
        plt.ylabel('Intensidade Normalizada')
        plt.grid(True)

        # Salvar graficos
        graphic = "porj_" + filename
        plt.savefig(graphic)
        plt.close()

    cv2.imwrite("saida.png", img)
    cv2.destroyAllWindows()


def define_rois():

    # Coordenadas pré estabelecidas para as áreas de interesse (ROI)
    roi_1 = (850, 660, 1560, 1110)
    roi_2 = (850, 1770, 570, 200)
    roi_3 = (70, 2070, 600, 250)
    roi_4 = (1330, 2070, 700, 220)
    roi_5 = (900, 2290, 1480, 220)

    # Adiciona os rois à lista global
    global rois
    rois.append(roi_1)
    rois.append(roi_2)
    rois.append(roi_3)
    rois.append(roi_4)
    rois.append(roi_5)


def main(input_dir, output_dir=None):
    
    # Verifica se o diretório de saída existe e criar se não existe
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista de resultados
    results = []

    # Processa os arqs no diretório de entrada
    for arquivo in os.listdir(input_dir):
        img_path = os.path.join(input_dir, arquivo)

        # Segmentar os formulários
        segment_form(img_path)

    

    # Ver se tem <dir_saida>
        # Se tiver então geramos as imgs de saida e o results.txt
        # Caso contrário so gera o results.txt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Comando esperado: forms.py <dir_entrada> <dir_saida>.")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    define_rois()
    main(input_dir, output_dir)
