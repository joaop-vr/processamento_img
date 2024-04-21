import sys
import os
import cv2
import numpy as np
from math import floor
import matplotlib.pyplot as plt



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


def segment_form(img_path, rois):

    # Carregar imagem
    img = cv2.imread(img_path)

    print(f"arquivo atual::{img_path}")

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

        if i == 0:
            soma = np.sum(histogram)
            print(f"soma:{soma}")

        # Normalizar o histograma
        histogram = histogram / np.max(histogram)

        if 6 <= i <= 8:
            middle_index = np.floor(len(histogram) / 2).astype(int)
            
            # Encontrar o índice do menor valor do histograma normalizado
            #min_magnitude_index = np.abs(histogram[2:-2]).argmin()
    
            # Retorna o valor do menor magnitude e o índice
            #min_magnitude_value = histogram[min_magnitude_index]

            # Extrai o campo central de cada tupla do histograma (índice 1)
            central_values = [tup[1] for tup in histogram[2:-2]]
            
            # Calcula os valores absolutos
            absolute_values = np.abs(central_values)
            
            # Encontra o índice do menor valor em magnitude
            min_magnitude_index = absolute_values.argmin()

            #for i in range(len(histogram)):
                #print(f"histogram[{i}]: {histogram[i]}")

            #print(f"middle: {middle_index} | min_index: {min_magnitude_index} | min_value: {central_values[min_magnitude_index]}")

            if min_magnitude_index <= middle_index:
                print(filename + " -> Yes\n")
            else:
                print(filename + " -> No\n")

        if i == 9:
            # Defina os índices de início e fim do histograma
            inicio_histo = 2
            fim_histo = -2
            histograma_cortado = histogram[inicio_histo:fim_histo]

            # Valor de guarda
            threshold_variancia = 100 

            # Divida o eixo X do histograma cortado em 10 intervalos iguais
            num_intervalos = 10
            intervalo_tamanho = len(histograma_cortado) // num_intervalos

            # Calcule a variância para cada intervalo no eixo X
            variâncias = []
            for i in range(num_intervalos):
                inicio = i * intervalo_tamanho
                fim = (i + 1) * intervalo_tamanho
                sub_histograma = histograma_cortado[inicio:fim]
                variância_atual = np.var(sub_histograma)
                
                # Aplicar o valor de guarda
                if variância_atual <= threshold_variancia:
                    variâncias.append(variância_atual)
                else:
                    variâncias.append(0)  # Ou outro valor que indique que o intervalo foi ignorado


            # Encontre o intervalo com a maior variância
            indice_max_variancia = np.argmax(variâncias)
            max_variancia = variâncias[indice_max_variancia]

            print(f"Variancias:{variâncias}\n")

            print(f"Última resposta da imagem {img_path}: {indice_max_variancia+1}\n")

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

    saida = img_path[-11:-4] + "_saida.png"
    cv2.imwrite(saida, img)


def define_rois_form_1():

    rois = []

    # Coordenadas pré estabelecidas para as áreas de interesse (ROI)
    roi_0 = (900, 20, 625, 170)
    roi_1 = (850, 760, 1560, 120)
    roi_11 = (850, 950, 1560, 110)
    roi_12 = (850, 1120, 1560, 120)
    roi_13 = (850, 1310, 1560, 120)
    roi_14 = (850, 1490, 1560, 120)
    roi_15 = (850, 1670, 1560, 90)
    roi_2 = (850, 1770, 570, 200)
    roi_3 = (70, 2070, 600, 250)
    roi_4 = (1330, 2070, 700, 220)
    roi_5 = (900, 2290, 1480, 220)


    # Adiciona os rois à lista global
    rois.append(roi_0)
    rois.append(roi_1)
    rois.append(roi_11)
    rois.append(roi_12)
    rois.append(roi_13)
    rois.append(roi_14)
    rois.append(roi_15)
    rois.append(roi_2)
    rois.append(roi_3)
    rois.append(roi_4)
    rois.append(roi_5)

    return rois


def define_rois_form_2():

    rois = []

    # Coordenadas pré estabelecidas para as áreas de interesse (ROI)
    roi_0 = (850, 700, 1560, 150)
    roi_1 = (850, 870, 1560, 150)
    roi_2 = (850, 1050, 1560, 150)
    roi_3 = (850, 1250, 1560, 150)
    roi_4 = (850, 1425, 1560, 150)
    roi_5 = (850, 1600, 1560, 150)
    roi_6 = (850, 1780, 590, 200)
    roi_7 = (70, 2110, 600, 200)
    roi_8 = (1330, 2120, 700, 170)
    roi_9 = (900, 2290, 1490, 220)

    # Adiciona os rois à lista global
    rois.append(roi_0)
    rois.append(roi_1)
    rois.append(roi_2)
    rois.append(roi_3)
    rois.append(roi_4)
    rois.append(roi_5)
    rois.append(roi_6)
    rois.append(roi_7)
    rois.append(roi_8)
    rois.append(roi_9)

    return rois


def remove_labels(img_path):

    # Carregar imagem
    img = cv2.imread(img_path)

    rois = []

    # Coordenadas pré estabelecidas para as áreas de interesse (ROI)
    roi_1 = (980, 700, 350, 1050)
    roi_2 = (1470, 700, 170, 1050)
    roi_3 = (1900, 700, 140, 1050)
    roi_4 = (2240, 700, 130, 1050)
    roi_5 = (990, 1820, 130, 120)
    roi_6 = (1350, 1820, 130, 120)
    roi_7 = (210, 2130, 100, 120)
    roi_8 = (560, 2130, 100, 120)
    roi_9 = (1480, 2130, 100, 120)
    roi_10 = (1830, 2130, 100, 120)

    # Adiciona os rois à lista global
    rois.append(roi_1)
    rois.append(roi_2)
    rois.append(roi_3)
    rois.append(roi_4)
    rois.append(roi_5)
    rois.append(roi_6)
    rois.append(roi_7)
    rois.append(roi_8)
    rois.append(roi_9)
    rois.append(roi_10)


    for i in range(len(rois)):

        # Variáveis delimitadoras da região de interesse (ROI)
        x, y, w, h = rois[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255, 0), 2)

        # Recortar região de interesse (ROI)
        roi = img[y:y+h, x:x+w]
    
        img[y-2:y+h+2, x-2:x+w+2] = 255


    saida = img_path[-11:-4] + "_labels_deleted.png"
    cv2.imwrite(saida, img)




def apartation(img_path, i):

    img = cv2.imread(img_path)

    x, y, w, h = (900, 20, 625, 170)
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)

    # Recortar região de interesse (ROI)
    roi = img[y:y+h, x:x+w]

    # Salvar o ROI
    filename = "roi"+str(i) +".png"
    cv2.imwrite(filename, roi)

    # Calcular o histograma ao longo do eixo X
    img_2 = cv2.imread(filename)
    histogram = np.sum(img_2, axis=0)

    soma = np.sum(histogram)
    print(f"soma:{soma}")
            
    if soma > 70000000:
        print("Ficha2: " + img_path)
    else:
        print("Ficha1: " + img_path)
    

def select_field_propotype(num_question, num_answer):

    roi_1a = (50, 50, 50, 5)

    # Variáveis delimitadoras da região de interesse (ROI)
    x, y, w, h = roi_1a
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255, 0), 2)

    cv2.imwrite(".png", img)

def select_field(img_path):

    roi_1a = (940, 860, 80, 5)

    roi_1b = (1400, 860, 80, 5)
    roi_1c = (1860, 860, 80, 5)

    roi_2a = (940, 1040, 80, 5)

    roi_2b = (1400, 1040, 80, 5)

    img = cv2.imread(img_path)

    # Variáveis delimitadoras da região de interesse (ROI)
    x, y, w, h = roi_1a
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)

    x, y, w, h = roi_1b
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)

    x, y, w, h = roi_1c
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
    x, y, w, h = roi_2a
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)
    x, y, w, h = roi_2b
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 2)

    cv2.imwrite("A_saida.png", img)


def main(rois, input_dir, output_dir=None):
    
    # Verifica se o diretório de saída existe e criar se não existe
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista de resultados
    results = []

    
    i = 0
    # Processa os arqs no diretório de entrada
    for arquivo in os.listdir(input_dir):

        i = i + 1
        img_path = os.path.join(input_dir, arquivo)

        # Carregar imagem
        img = cv2.imread(img_path)

        apartation(img_path, i)

        # Binarizar a imagem
        #_, imagem_binarizada = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        #img_path_2 = img_path[-11:-4] + "_binary.png"
        #cv2.imwrite(img_path_2, imagem_binarizada)

        # Remove os labels
        remove_labels(img_path)

        # Segmentar os formulários
        #segment_form(img_path, rois)
        select_field(img_path)



    # Ver se tem <dir_saida>
        # Se tiver então geramos as imgs de saida e o results.txt
        # Caso contrário so gera o results.txt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Comando esperado: forms.py <dir_entrada> <dir_saida>.")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    rois = define_rois_form_2()
    main(rois, input_dir, output_dir)
