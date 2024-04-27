import cv2
import numpy as np
import sys

def define_rois(id, x, y, img, img_path):

    # Lista de áreas de interesse
    rois = []

    print(x,y)

    if id == 1:
        # Coordenadas pré estabelecidas para as áreas de interesse (ROI) do Form 1
        roi_1 = (x+832, y-260, 1560, 100) #
        roi_2 = (x+832, y-90, 1560, 110)
        roi_3 = (x+832, y+90, 1560, 120)
        roi_4 = (x+832, y+280, 1560, 120)
        roi_5 = (x+832, y+470, 1560, 120)
        roi_6 = (x+832, y+650, 1560, 90)
        roi_7 = (x+832, y+810, 590, 140)
        roi_8 = (x+90, y+1130, 500, 150)
        roi_9 = (x+1350, y+1140, 500, 150)
        roi_10 = (x+850, y+1300, 1520, 150)
    else:
        # Coordenadas pré estabelecidas para as áreas de interesse (ROI) do Form 2
        roi_1 = (x+722, y+169, 1560, 150)
        roi_2 = (x+722, y+339, 1560, 150)
        roi_3 = (x+722, y+519, 1560, 150)
        roi_4 = (x+722, y+719, 1560, 150)
        roi_5 = (x+722, y+894, 1560, 150)
        roi_6 = (x+722, y+1069, 1560, 150)
        roi_7 = (x+722, y+1249, 590, 200)
        roi_8 = (70, y+1569, 600, 200)
        roi_9 = (1330, y+1569, 700, 170)
        roi_10 = (900, y+1759, 1490, 220)

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

    

    return rois

def define_threshold(img):

    # Calcula o histograma no eixo Y
    histogram_Y = np.sum(img, axis=1)

    # Normaliza o histograma
    histogram_Y = histogram_Y / np.max(histogram_Y)

    # Extrai o campo central de cada tupla do histograma (índice 1)
    #central_values_Y = [tup[1] for tup in histogram_Y]
    
    # Calcula os valores absolutos
    #absolute_values_Y = np.abs(central_values_Y)

    # Lista dos indices dos mínimos locais
    idx_mins = []
    
    for i in range(0, len(histogram_Y)//2):
        if histogram_Y[i] < histogram_Y[i-1] and histogram_Y[i] < histogram_Y[i+1]:
            idx_mins.append(i)
            
    # Ordenar de forma decrescente os índices dos mínimos locais
    idx_mins.sort(key=lambda x: histogram_Y[x])

    # Calcula as coordenadas do roi referente ao marco zero 
    # (referencial para estabelecer as demais áreas de interesse)
    idx_middle = (idx_mins[0]+idx_mins[1])//2
    x = 0
    y = idx_middle - 80
    w = img.shape[1]-1
    h = 160

    roi = img[y:y+h, x:x+w]
    _, binarized_roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

    # Variáveis para armazenar os índices do primeiro pixel preto
    first_black_y = None
    first_black_x = None

    # Percorra a ROI binarizada para encontrar o primeiro pixel preto
    for i in range(roi.shape[0]): 
        for j in range(roi.shape[1]):  
            if (binarized_roi[i, j] == 0).any():  # Se o pixel é preto
                first_black_y = i + y 
                first_black_x = j + x 
                break  # Sai do loop interno se encontrar o primeiro pixel preto
        if first_black_y is not None and first_black_x is not None:
            break  # Sai do loop externo se encontrar o primeiro pixel preto

    if first_black_y is None or first_black_x is  None:
        print("Não foi possível encontrar um pixel preto.")

    return first_black_x, first_black_y



# Carregar a imagem
image = cv2.imread(sys.argv[1])

# Converter para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

x, y = define_threshold(gray)

rois = define_rois(1, x, y, gray, "Antes do recorte.png")

recort = gray[y:y+2000, :]

cv2.imwrite("Recorte.png", recort)
# Detectar as bordas usando o operador de Canny
edges = cv2.Canny(recort, 50, 150, apertureSize=3)
cv2.imwrite('Edges.png', edges)

# Aplicar a transformada de Hough para detecção de linhas
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=5)

# Identificar linhas horizontais
horizontal_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
    if abs(angle) < 10 or abs(angle - 180) < 10:
        horizontal_lines.append(line)

# Calcular a média dos ângulos das linhas horizontais
mean_angle = np.mean([np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi for line in horizontal_lines])

# Corrigir a rotação da imagem
(h, w) = recort.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -mean_angle, 1.0)
rotated_image = cv2.warpAffine(recort, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

rois = define_rois(1, x, y, rotated_image, "Depois do recorte.png")

# Mostrar os resultados
cv2.imwrite('Original Image.png', gray)
cv2.imwrite('Rotated Image.png', rotated_image)

for i in range(len(rois)):
    x, y, w, h = rois[i]
    roi = rotated_image[y:y+h, x:x+w]
    cv2.rectangle(rotated_image, (x,y), (x+w, y+h), (0,255, 0), 2)

filename = "Saida1_" + sys.argv[1][:-4].split('/')[-1] + ".png"
    
cv2.imwrite(filename, rotated_image)
