# README

## Projeto: Classificação de Imagens de Textura

### Descrição do Projeto
Este projeto tem como objetivo realizar a classificação de imagens de textura utilizando a base de dados FSD-M, especificamente o arquivo [`macroscopic0.zip`](https://zenodo.org/records/10219797), que contém 557 imagens distribuídas em 9 classes. A classe da imagem está codificada no nome do arquivo. O projeto envolve a extração de diversas representações das imagens e a aplicação do algoritmo k-NN com k=1 para a classificação. Os resultados obtidos foram analisados e interpretados.

### Estrutura do Projeto
- **classificacao_textura.py**: O script principal que realiza a extração de características, o particionamento dos dados, o treinamento e a classificação.
- **Imagens**: Arquivo [`macroscopic0.zip`](https://zenodo.org/records/10219797) contendo as imagens de textura a serem classificadas.
- **Relatório**: Arquivo PDF "Relatorio - Lab5" detalhando os experimentos realizados e os resultados obtidos.

### Entrada e Saída
- **Entrada**: 
  - Imagens de textura na pasta [`macroscopic0.zip`](https://zenodo.org/records/10219797).
- **Saída**: 
  - Resultados da classificação.

### Execução do Programa
Para executar o programa, utilize o seguinte comando:
```bash
> python3 <código_para_representação> <diretorio_imagens>
