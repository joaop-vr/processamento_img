# README

## Projeto: Segmentação de Áreas de Floresta com Base na Cor Verde

### Descrição do Projeto
Este projeto envolve a implementação de um programa em Python que segmenta áreas de floresta usando exclusivamente a informação de cor. O objetivo é identificar e realçar as áreas verdes em imagens coloridas, utilizando o modelo de cor mais apropriado para essa tarefa.

### Estrutura do Projeto
- **floresta.py**: O script principal que realiza a segmentação de áreas verdes nas imagens.
- **Imagens de Teste**: Imagens fornecidas para testar a qualidade da segmentação.

### Entrada e Saída
- **Entrada**: 
  - Imagem de entrada (nome do arquivo da imagem a ser processada).
  - Imagem de saída (nome do arquivo onde a imagem segmentada será salva).
- **Saída**: 
  - Imagem segmentada realçando a área de floresta (verde).

### Execução do Programa
Para executar o programa, utilize o seguinte comando:
```bash
> python3 floresta.py <imagem_entrada> <imagem_saida>
```

## Apontamentos:

 - Optei por usar o modelo de cor HSV;
 - Pesquisei formas de melhorar a máscara, por exemplo erosão, dilatação,
fechamento, abertura e fechamento&abertura. Os melhores resultados foram obtidos 
pela dilatação, sendo aplicado uma única iteração.

 - Passo a passo:
    1) Determinação da gama de valores para o verde
    2) Determinação do kernel
    3) Aplicação do objeto kernel na dilatação
    4) Aplicação de máscara
    