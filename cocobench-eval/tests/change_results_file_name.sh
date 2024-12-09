for file in result_*_*.jsonl; do
    if [[ "$file" =~ ^result_(CG|CM|CR|CUF|CUR)_[0-9]+_[0-9]+\.jsonl$ ]]; then
        new_file="${file%.jsonl}_512.jsonl"
        mv "$file" "$new_file"
        echo "Renamed: $file -> $new_file"
    fi
done