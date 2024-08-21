# README

## Projeto: Segmentação de Formulários de Satisfação do Hotel

### Descrição do Projeto
Este projeto consiste em desenvolver um programa em Python que processa formulários de satisfação preenchidos por hóspedes de um hotel no Paquistão. O objetivo é extrair as respostas de 10 perguntas específicas desses formulários e converter essas informações em um formato que permita a geração de estatísticas.

### Estrutura do Projeto
- **forms_segmentacao_manual.py**: O script principal que realiza todo o processamento de forma a segmentar as imagens manualmente.
- **forms_segmentacao_relativa.py**: O script principal que realiza todo o processamento de forma a segmentar as imagens a partir de um marco-zero comum a todas imagens.
- **Archive.zip**: Imagens de formulários de exemplo que são utilizadas para testar o programa.

### Requisitos
1. **Processamento de Formulários**: O programa deve processar todos os formulários do diretório de entrada.
2. **Saída em Arquivo Texto**: A saída do programa deve ser um arquivo texto (ASCII) contendo 10 linhas, cada uma representando a porcentagem de respostas para cada pergunta. A última linha deve mostrar a média das notas dos clientes.
3. **Geração de Imagens Marcadas**: Opcionalmente, o programa deve gerar imagens marcadas indicando os campos lidos durante o processamento. Essas imagens devem ser salvas em um diretório de saída especificado.

### Entrada e Saída
- **Entrada**: Diretório contendo os formulários (imagens) a serem processados.
- **Saída**: 
  - Arquivo texto `results.txt` com as porcentagens de respostas e a média das notas dos clientes.
  - Diretório opcional com imagens marcadas.

### Execução do Programa
Para executar o programa, utilize o seguinte comando:
```bash
> python <script_de_segmentação> <diretorio_entrada> <diretorio_saida>