import sys
import os
import cv2
import numpy as np
from math import floor
import matplotlib.pyplot as plt

def listar_arquivos(diretorio):

    i = 0
    for arquivo in os.listdir(diretorio):

        print(f"arquivo:{arquivo}")
        # Variáveis delimitadoras da região de interesse (ROI)
        img = cv2.imread(arquivo)
        x, y, w, h = (900, 20, 625, 170)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 255), 2)

        # Recortar região de interesse (ROI)
        rois
        roi = img[y:y+h, x:x+w]

        # Salvar o ROI
        filename = "roi"+str(i) +".png"
        cv2.imwrite(filename, roi)

        # Calcular o histograma ao longo do eixo X
        img_2 = cv2.imread(filename)
        histogram = np.sum(img_2, axis=0)

        if i == 0:
            soma = np.sum(histogram)
            print(f"soma:{soma}")

        if soma > 70000000:
            print("Ficha2: " + arquivo)
        else:
            print("Ficha1: " + arquivo)
        
        i = i + 1


if __name__ == "__main__":
    # Verifica se um diretório foi passado como argumento
    if len(sys.argv) < 2:
        print("Por favor, passe o diretório como argumento.")
        sys.exit(1)

    # Pega o diretório do primeiro argumento
    diretorio = sys.argv[1]

    listar_arquivos(diretorio)

    
