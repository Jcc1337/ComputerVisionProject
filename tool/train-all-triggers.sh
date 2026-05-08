#!/bin/bash

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate Computervision28
PYTHON=python

dataset=$1
trigger_folder=${2:-trigger}  # Default to 'trigger' folder if not specified
exp_name=${3:-deeplabv3}      # Default to 'deeplabv3' if not specified

if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset> [trigger_folder] [exp_name]"
    echo "Example: $0 cityscapes trigger deeplabv3"
    exit 1
fi

if [ ! -d "$trigger_folder" ]; then
    echo "Error: Trigger folder '$trigger_folder' not found"
    exit 1
fi

# Get all image files
image_files=$(find "$trigger_folder" -maxdepth 1 -type f \( -iname "*.png" -o -iname ".jpg" -o -iname "*.jpeg" \))

if [ -z "$image_files" ]; then
    echo "Error: No image files found in '$trigger_folder'"
    exit 1
fi

trigger_count=0
for image_path in $image_files; do
    trigger_name=$(basename "$image_path" | sed 's/\.[^.]*$//')  # Extract filename without extension
    echo "=========================================="
    echo "Training with trigger: $trigger_name"
    echo "Image: $image_path"
    echo "=========================================="
    
    # Update config file with new trigger settings
    config="config/${dataset}/${dataset}_${exp_name}.yaml"
    if [ ! -f "$config" ]; then
        echo "Error: Config file not found: $config"
        exit 1
    fi
    
    # Update trigger_name and trigger_path in config
    sed -i "s|trigger_name: .*|trigger_name: ${trigger_name}|" "$config"
    sed -i "s|trigger_path: .*|trigger_path: ${image_path}|" "$config"
    
    # Call the original training script
    bash tool/train.sh "$dataset" "$exp_name"
    
    if [ $? -ne 0 ]; then
        echo "Training failed for trigger: $trigger_name"
        exit 1
    fi
    
    trigger_count=$((trigger_count + 1))
    echo "Completed training $trigger_count/$( echo "$image_files" | wc -w )"
    echo ""
done

echo "=========================================="
echo "All $trigger_count training runs completed!"
echo "==========================================