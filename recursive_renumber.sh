#!/bin/bash

# Function to renumber .tif files
renumber_tif() {
    local tif_files=$(ls *.tif 2>/dev/null)

    if [ -n "$tif_files" ]; then
        for file in $tif_files; do
            base_name=$(basename "$file" .tif)
            number=$(echo "$base_name" | cut -d'_' -f2)
            new_name="${base_name%%_*}_$(printf "%05d" $number).tif"
            mv "$file" "$new_name"
        done
    fi
}

# Recursive function to apply renumber_tif in subdirectories
recurse_renumber() {
    local directories=($(find . -type d))

    for dir in "${directories[@]}"; do
        if [ ! -f "$dir/.visited" ]; then
            touch "$dir/.visited"
            pushd "$dir" > /dev/null || continue

            # Print the current directory for debugging
            echo "Processing directory: $dir"

            renumber_tif
            popd > /dev/null || exit
        fi
    done
}

# Start in the specified folder or current folder if not specified
target_folder="$1"
if [ -n "$target_folder" ]; then
    cd "$target_folder" || exit
fi

# Print the starting directory for debugging
echo "Starting in directory: $(pwd)"

# Start the renumbering process
renumber_tif
recurse_renumber