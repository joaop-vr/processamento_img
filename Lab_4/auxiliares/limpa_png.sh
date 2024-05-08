for file in *.png; do
    if [[ ${#file} -gt 14 ]]; then
        rm "$file"
    fi
done
