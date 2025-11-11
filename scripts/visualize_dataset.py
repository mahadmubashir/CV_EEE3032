"""
Dataset Classes Visualization Script
EEE3032 - Computer Vision and Pattern Recognition

This script visualizes the MSRC dataset by:
1. Showing example images from 5 different classes
2. Displaying a table with the number of images per class (all 20 classes)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
OUTPUT_FOLDER = './experiments/dataset_visualization'

# Number of example classes to show
N_EXAMPLE_CLASSES = 5


def get_class_id_from_filename(filename):
    """
    Extract the integer class ID from an image filename.
    
    Args:
        filename: Image filename (e.g., '12_33_s.bmp')
    
    Returns:
        int: Class ID (e.g., 12) or -1 if parsing fails
    """
    try:
        return int(filename.split('_')[0])
    except (ValueError, IndexError):
        return -1


def analyze_dataset_classes(dataset_folder):
    """
    Analyze the dataset to extract all classes and count images per class.
    
    Args:
        dataset_folder: Path to the dataset Images folder
    
    Returns:
        dict: Dictionary with:
            - 'classes': sorted list of all class IDs
            - 'class_counts': dict mapping class_id -> count
            - 'class_examples': dict mapping class_id -> example image filename
            - 'total_images': total number of images
    """
    print("="*80)
    print("ANALYZING DATASET")
    print("="*80)
    
    # Get all image files
    image_files = [f for f in os.listdir(dataset_folder) if f.endswith('.bmp')]
    total_images = len(image_files)
    
    # Extract classes and count images per class
    class_counts = defaultdict(int)
    class_examples = {}  # Store first image found for each class
    
    for filename in sorted(image_files):
        class_id = get_class_id_from_filename(filename)
        if class_id != -1:
            class_counts[class_id] += 1
            # Store first image as example for this class
            if class_id not in class_examples:
                class_examples[class_id] = filename
    
    # Get sorted list of classes
    classes = sorted(class_counts.keys())
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images: {total_images}")
    print(f"   Number of classes: {len(classes)}")
    print(f"\n📋 Images per class:")
    for class_id in classes:
        print(f"   Class {class_id:2d}: {class_counts[class_id]:3d} images")
    
    print("="*80)
    
    return {
        'classes': classes,
        'class_counts': dict(class_counts),
        'class_examples': class_examples,
        'total_images': total_images
    }


def visualize_example_images(dataset_info, save_path):
    """
    Create a visualization showing example images from 5 different classes.
    
    Args:
        dataset_info: Dictionary from analyze_dataset_classes()
        save_path: Path to save the visualization
    """
    classes = dataset_info['classes']
    class_counts = dataset_info['class_counts']
    class_examples = dataset_info['class_examples']
    
    # Select 5 example classes (evenly distributed)
    n_classes = len(classes)
    if n_classes >= N_EXAMPLE_CLASSES:
        # Select evenly spaced classes
        step = n_classes // N_EXAMPLE_CLASSES
        example_class_indices = [i * step for i in range(N_EXAMPLE_CLASSES)]
        example_classes = [classes[i] for i in example_class_indices]
    else:
        # Use all available classes
        example_classes = classes[:N_EXAMPLE_CLASSES]
    
    print(f"\n📷 Selected {len(example_classes)} example classes: {example_classes}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 4))
    
    for idx, class_id in enumerate(example_classes):
        example_filename = class_examples[class_id]
        example_path = os.path.join(DATASET_FOLDER, example_filename)
        
        # Load and display image
        ax = fig.add_subplot(1, len(example_classes), idx + 1)
        img = cv2.imread(example_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f'Class {class_id}\n({class_counts[class_id]} images)', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'Class {class_id}\nImage not found', 
                   ha='center', va='center', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Example Images from 5 Different Classes\nMSRC Object Category Image Database v2', 
                fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Example images visualization saved to: {save_path}")
    plt.close()


def visualize_class_statistics(dataset_info, save_path):
    """
    Create a visualization showing class statistics table.
    
    Args:
        dataset_info: Dictionary from analyze_dataset_classes()
        save_path: Path to save the visualization
    """
    classes = dataset_info['classes']
    class_counts = dataset_info['class_counts']
    class_examples = dataset_info['class_examples']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Create table data
    table_data = []
    table_data.append(['Class ID', 'Number of Images', 'Example Image'])
    
    for class_id in classes:
        count = class_counts[class_id]
        example = class_examples.get(class_id, 'N/A')
        table_data.append([
            f'{class_id}',
            f'{count}',
            example
        ])
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.15, 0.25, 0.60])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows (alternating colors)
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Dataset Class Statistics (All 20 Classes)\nMSRC Object Category Image Database v2', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Class statistics table saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to generate dataset visualization.
    """
    print("="*80)
    print("DATASET CLASSES VISUALIZATION")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Analyze dataset
    dataset_info = analyze_dataset_classes(DATASET_FOLDER)
    
    # Generate two separate visualizations
    # Visualization 1: Example images from 5 classes
    example_images_path = os.path.join(OUTPUT_FOLDER, 'dataset_example_images.png')
    visualize_example_images(dataset_info, example_images_path)
    
    # Visualization 2: Class statistics table
    statistics_path = os.path.join(OUTPUT_FOLDER, 'dataset_class_statistics.png')
    visualize_class_statistics(dataset_info, statistics_path)
    
    print("\n" + "="*80)
    print("✅ DATASET VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n📁 Generated files:")
    print(f"   1. {example_images_path}")
    print(f"   2. {statistics_path}")
    print("="*80)


if __name__ == "__main__":
    main()

