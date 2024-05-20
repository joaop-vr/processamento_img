# README

## Projeto: Comparação de Filtros para Remoção de Ruído "Salt and Pepper"

### Descrição do Projeto
O objetivo deste projeto é testar e comparar diferentes tipos de filtros para a remoção de ruído "salt and pepper" em imagens. Utilizaremos uma imagem de referência e adicionaremos diferentes níveis de ruído. Os filtros a serem comparados são: cvBlur, cvGaussianBlur, cvMedianBlur e Stacking. A performance dos filtros será avaliada usando a métrica PSNR (Peak Signal-to-Noise Ratio).

### Estrutura do Projeto
- **filtros.py**: O script principal que adiciona ruído às imagens, aplica os filtros e calcula o PSNR.
- **função de adição de ruído**: Código fornecido para adicionar ruído "salt and pepper" às imagens.
- **Imagem de referência**: Imagem em anexo para os testes.

### Entrada e Saída
- **Entrada**: 
  - Imagem de referência.
  - Níveis de ruído: [0.01, 0.02, 0.05, 0.07, 0.1].
- **Saída**: 
  - Imagens filtradas.
  - Valores de PSNR para cada filtro e nível de ruído.

### Execução do Programa
Para executar o programa, utilize o seguinte comando:
```bash
> python3 filtros.py <imagem_entrada> <diretorio_saida>
```

## Apontamentos:

 - Para rodar o programa é preciso dar o seguinte comando: $python3 remove_ruidos.py <img_entrada>;

 - Não há geração de imagem ao final do programa;

 -  O script graph.py, dentro da pasta "auxiliares", é responsável por gerar imagens presentes no relatório a partir de arquivos .txt que foram utilizados como entrada.