# README

## Projeto: Comparação de Imagens Utilizando Histogramas

### Descrição do Projeto
Este projeto envolve a implementação de um programa em Python que compara imagens de cinco personagens diferentes utilizando histogramas. O objetivo é comparar cada uma das 25 imagens com as 24 restantes usando diferentes métodos de comparação de histogramas e calcular a acurácia e a matriz de confusão para cada método.

### Estrutura do Projeto
- **histograma.py**: O script principal que realiza a comparação dos histogramas.
- **Imagens**: O arquivo `Archive.zip` contém 25 imagens de 5 personagens diferentes (cinco imagens de cada personagem).

### Métodos de Comparação Implementados
1. **Distância Euclidiana**: Comparação utilizando a distância euclidiana entre histogramas.
2. **CV_COMP_CORREL**: Correlação entre histogramas.
3. **CV_COMP_CHISQR**: Chi-Square entre histogramas.
4. **CV_COMP_INTERSECT**: Interseção entre histogramas.
5. **CV_COMP_BHATTACHARYYA**: Distância de Bhattacharyya entre histogramas.

### Entrada e Saída
- **Entrada**: 
  - Método de cálculo (número de 1 a 5)
  - Diretório contendo as imagens a serem processadas.
- **Saída**: 
  - Acurácia
  - Matriz de confusão

### Execução do Programa
Para executar o programa, utilize o seguinte comando:
```bash
> python histograma.py <metodo> <diretorio_imagens>
```
## Apontamentos:

 - Optei por implementar a distância euclidiana e por utilizar as demais
métricas (correlação, interseção, etc...) por meio de parâmetros da função
"compHist";

 - Implementei a classe RegistroImagem para facilitar na manipulação
de dados;

 - Passo a passo:
    1) Calculei três histogramas para cada imagem (azul, verde e vermelho);
    2) Comparei os histogramas azul, verde e vermelho de cada imagem com os 
    respectivos histogramas azul, verde e vermelho das demais 24 imagens. O
    resultado foi armazenado no vetor "comparacoes" presente na classe RegistroImagem;
    3) Optei por interpretar cada canal de cor como sendo uma variável de um vetor em R3,
    pois dessa forma poderia calcular a norma de cada vetor e tratar as normas como 
    sendo o valor dos vizinhos;
    4) Ordenei o vetor de normas, peguei os k-vizinhos (k=3) e classifiquei a imagem pivo
    como sendo da classe de maior frequência dentre os k-vizinhos;