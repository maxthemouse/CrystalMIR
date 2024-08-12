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

# Function to apply renumber_tif in each directory found
list_renumber() {
    # Find all directories and process each one
    find . -type d | while read -r dir; do
        echo "Processing directory: $dir"
        (cd "$dir" && renumber_tif)
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
list_renumber
