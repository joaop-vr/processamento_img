import sys
import glob
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class RegistroImagem:
    def __init__(self, nome_imagem, hist_imagem):
        self.nome_imagem = nome_imagem
        self.hist_imagem = hist_imagem


def aplicar_knn(X_data, y_data):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_data, y_data)
    y_pred = knn.predict(X_data)
    return y_pred



def aplicar_metodo(metodo, registros):
    n = len(registros)

    # Testar
    print("N:", n)
    mat_similariade = []
    
    for i in range(n):
        hist_pivo = registros[i].hist_imagem
        print("Nome imagem pivo:" + registros[i].nome_imagem)
        linha = []
        linha.append(registros[i].nome_imagem)
        # mat_similariade.append(registros[i].nome_imagem)
        for j in range(n):
            if i != j:
                hist_iterado = registros[j].hist_imagem
                comparacao = cv2.compareHist(hist_pivo, hist_iterado, metodo)
                linha.append(comparacao)
                #matriz_sim[i][j] = cv2.compareHist(hist_pivo, hist_iterado, metodo)
        mat_similariade.append(linha)

    #Testar
    print("mat_similariade::")
    for linha in mat_similariade:
        for elemento in linha:
            print(elemento)
        print()
        print()
        

    return mat_similariade


def metodos_comparacao(metodo_id):
    if metodo_id == 1:
        metodo = dist_euclidiana
    elif metodo_id == 2:
        metodo = cv2.HISTCMP_CORREL
    elif metodo_id == 3:
        metodo = cv2.HISTCMP_CHISQR
    elif metodo_id == 4:
        metodo = cv2.HISTCMP_INTERSECT
    elif metodo_id == 5:
        metodo = cv2.HISTCMP_BHATTACHARYYA
    else:
        print("Código inserido é inválido!")
        sys.exit(1)

    return metodo
    


def criar_registros(dir):
    arquivos = glob.glob(dir + "/*.bmp")

    # Testar
    print("arquivos:")
    for arquivo in arquivos:
        print(arquivo)

    n = len(arquivos)
    registros = []
    
    for arquivo in arquivos:

        # Estabelecemos a imagem pivo, convertemos para cinza,
        # montamos o histograma e normalizamos
        imagem = cv2.imread(arquivo)
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([imagem_cinza], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Istanciando o registro para posteriormente
        # adicioná-lo em um array (registros)
        registro_aux = RegistroImagem(arquivo, hist)
        registros.append(registro_aux)

    return registros

def main():
    if len(sys.argv) < 3:
        print("Comando esperado: histograma.py <metodo_de_calculo> <diretorio_de_imagens>.")
        sys.exit(1)

    dir = sys.argv[2]
    metodo_id = int(sys.argv[1])

    registros = criar_registros(dir)
    metodo = metodos_comparacao(metodo_id)
    mat_similaridade = aplicar_metodo(metodo, registros)

    # Aplicar KNN
    X_data = mat_similaridade[:,1:]
    y_data = mat_similariade[:,0]
    y_pred = aplicar_knn(X_data, y_data)

    # Calcular acurácia e matriz de confusão
    acuracia = sum(y_pred == y_data) / len(y_data)
    print("Acurácia:", acuracia)

    matriz_confusao = confusion_matrix(y_data, y_pred)
    print("Matriz de Confusão:")
    print(matriz_confusao)

    # Se desejar, também pode imprimir um relatório de classificação
    # print(classification_report(y_data, y_pred))

if __name__ == "__main__":
    main()
