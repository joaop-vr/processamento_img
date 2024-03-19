import sys
import glob
import cv2
import numpy as np

def aplicar_metodo(metodo, histogramas):

    # "rotulos": array com os personagens (5 personagens para o caso dos Simpsons)
    rotulos = []
    rotulos = glob.glob(dir+"/*1.bmp")

    # Teste
    print("rotulos: "+rotulos)

    # Matriz de similaridade
    print("len(histogramas): "+len(histogramas))
    matriz_sim = np.zeros((len(histogramas)), (len(histogramas)))
    
    if metodo == "DIST_EUCLID":
        # função para calcular distancia euclidiana
    else:
        for i in histogramas:
            hist_pivo = histogramas[i]
            for j in dir:
                if i != j:
                    hist_iterado = histogramas[j]
                    matriz_sim[i][j] = cv2.compareHist(hist_pivo, hist_iterado, metodo)

    # Aplicar o kNN logo
                    



def metodos_comparacao(metodo_id, histogramas):

    if metodo_id == 1:
        metodo = DIST_EUCLID
    elif metodo_id == 2:
        metodo = CV_COMP_CORREL
    elif metodo_id == 3:
        metodo = CV_COMP_CHISQR
    elif metodo_id == 4:
        metodo = CV_COMP_INTERSECT
    elif metodo_id == 5:
        metodo = CV_COMP_BHATTACHARYYA
    else:
        print("Código inserido é inválido!")

    aplicar_metodo(metodo, dir, histogramas)


def calcula_hist(dir):

    arquivos = glob.glob(dir+"/*.bmp")
    print(arquivos)

    histogramas = []

    # Estabelecemos a imagem pivo, convertemos para cinza,
    # montamos o histograma e normalizamos
    for imagem in arquivos:
        imagem_pivo = cv2.imread(arquivos[i])
        imagem_pivo_cinza = cv2.cvtColor(imagem_pivo, cv2.COLOR_BGR2GRAY)
        hist_pivo = cv2.calcHist([imagem_pivo_cinza], [0], None, [256], [0, 256])
        cv2.normalize(hist_pivo, hist_pivo, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        histogramas.append(hist_pivo)

    return histogramas

def main():

    # Verificar se foram passados os argumentos na linha de comando
    if len(sys.argv) < 3:
        print("Comando esperado: histograma. py <metodo_de_calculo> <diretorio_de_imagens>.")
        sys.exit(1)

    dir = sys.argv[2]
    metodo_id = int(sys.argv[1])

    # Teste 
    print(sys.argv)
    
    histogramas = calcula_hist(dir)
    metodos_comparacao(metodo_id, dir, histogramas)


if __name__ == "__main__":
    main()