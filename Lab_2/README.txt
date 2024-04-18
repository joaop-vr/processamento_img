#######################################

Aluno: João Pedro Vicente Ramalho
GRR: 20224169

#######################################

Apontamentos:

-> Optei por usar o modelo de cor HSV;
-> Pesquisei formas de melhorar a máscara, por exemplo erosão, dilatação,
fechamento, abertura e fechamento&abertura. Os melhores resultados foram obtidos 
pela dilatação, sendo aplicado uma única iteração.

-> Passo a passo:
    1) Determinação da gama de valores para o verde
    2) Determinação do kernel
    3) Aplicação do objeto kernel na dilatação
    4) Aplicação de máscara
    