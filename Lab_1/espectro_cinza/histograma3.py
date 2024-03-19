import sys
import glob
import cv2
import re
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



def aplicar_metodo(metodo, registros):
    n = len(registros)
    mat_similaridade = np.zeros((n, n))
    
    for i in range(n):
        hist_pivo = registros[i].hist_imagem
        for j in range(n):
            if i != j:
                hist_iterado = registros[j].hist_imagem
                mat_similaridade[i][j] = cv2.compareHist(hist_pivo, hist_iterado, metodo)
    
    return mat_similaridade


def metodos_comparacao(metodo_id):
    if metodo_id == 1:
        metodo = dist_euclidiana    # Meramente para identificação (vou implementar na mão)
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
    N = len(registros)
    K = 3
    y_pred = []
    for i in range(N):
        comparacoes = []
        for j in range(N):
            comparacoes.append(mat_similaridade[i][j])

        if metodo_id == 2 or metodo_id == 4:    # Métricas: correlação e interseção
            comparacoes.sort(reverse=True)
        else:                                   # Métricas: Chi Sqr, Bhattacharyya e Dist. Euclidiana
            comparacoes.sort()

        k_vizinhos = comparacoes[0:K]
        
        print("Amostra:", registros[i].nome_imagem)
        print("K-vizinhos:", k_vizinhos)

    # Pega o mais frequente dos escolhidos
    # classes_vizinhos = [registros[j].nome_imagem[0] for j in range(N) if j in k_vizinhos]
    # print("Classes dos vizinhos:", classes_vizinhos)

    # classe_escolhida = mode(classes_vizinhos)
    # y_pred.append(classe_escolhida)
    # print("Classe escolhida:", classe_escolhida)


    # agora faz y_data pegando as classes certas
    # tipo, o nome da imagem é b5, a classe é b (atribui um valor a essa classe)
    # tipo, o nome da imagem é l2, a classe é l (atribui um valor a essa classe)
    # tipo, o nome da imagem é h1, a classe é h (atribui um valor a essa classe)
    # tipo, o nome da imagem é mg3, a classe é mg (atribui um valor a essa classe)
    # tipo, o nome da imagem é m5, a classe é m (atribui um valor a essa classe)
    # tenha em mente que serão 5 classes, sendo que cada classe possui 5 imagens, então y_data vai ficar com tamanho 25
    y_data = []
    for registro in registros:
        # Apartar o nome da imagem usando "/" como separador e pegar a última parte
        partes_nome = registro.nome_imagem.split("/")
        nome = partes_nome[-1]

        # Excluir a extensão e os números
        nome_sem_extensao = nome.split(".")[0]
        nome_apenas_caracteres = re.findall("[a-zA-Z]", nome_sem_extensao)
        classe = "".join(nome_apenas_caracteres)
        y_data.append(classe)

    print("y_data::")
    print(y_data)

    print("y_pred::")
    print(y_pred)

    # Calcular acurácia e matriz de confusão
    #acuracia = sum(y_pred == y_data) / len(y_data)
    #print("Acurácia:", acuracia)

    #matriz_confusao = confusion_matrix(y_data, y_pred)
    #print("Matriz de Confusão:")
    #print(matriz_confusao)

if __name__ == "__main__":
    main()
