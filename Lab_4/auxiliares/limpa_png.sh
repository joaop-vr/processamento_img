# Script para apagar as imagens geradas como saida dos códigos para formulário de segmentação

for file in *.png; do
    if [[ ${#file} -gt 14 ]]; then
        rm "$file"
    fi
done
