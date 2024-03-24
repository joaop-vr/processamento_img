#######################################

Aluno: João Pedro Vicente Ramalho
GRR: 20224169

#######################################

Apontamentos:

-> Optei por implementar a distância euclidiana e por utilizar as demais
métricas (correlação, interseção, etc...) por meio de parâmetros da função
"compHist";

-> Implementei a classe RegistroImagem para facilitar na manipulação
de dados;

-> Passo a passo:
    1) Calculei três histogramas para cada imagem (azul, verde e vermelho);
    2) Comparei os histogramas azul, verde e vermelho de cada imagem com os 
    respectivos histogramas azul, verde e vermelho das demais 24 imagens. O
    resultado foi armazenado no vetor "comparacoes" presente na classe RegistroImagem;
    3) Optei por interpretar cada canal de cor como sendo uma variável de um vetor em R3,
    pois dessa forma poderia calcular a norma de cada vetor e tratar as normas como 
    sendo o valor dos vizinhos;
    4) Ordenei o vetor de normas, peguei os k-vizinhos (k=3) e classifiquei a imagem pivo
    como sendo da classe de maior frequência dentre os k-vizinhos;