"""
Comprehensive Evaluation Script
EEE3032 - Computer Vision and Pattern Recognition

This script evaluates multiple descriptor-distance combinations:
- Global RGB Histogram + Euclidean Distance
- Global RGB Histogram + Manhattan Distance
- Spatial Grid RGB Histogram + Euclidean Distance
- Spatial Grid RGB Histogram + Manhattan Distance
- Global RGB Histogram (PCA 4) + Mahalanobis Distance
- Spatial Grid RGB Histogram (PCA 4) + Mahalanobis Distance

Saves PR curves and precision/recall data for comparison.
"""

import os
import numpy as np
import json
from cvpr_evaluate import (
    analyze_dataset,
    evaluate_all_queries,
    plot_average_pr_curve
)
from cvpr_visualsearch import (
    load_descriptors,
    load_pca_model_for_descriptor
)
from cvpr_compare import (
    euclidean_distance,
    manhattan_distance,
    mahalanobis_distance
)

# Configuration
DESCRIPTOR_FOLDER = './descriptors'
DATASET_FOLDER = './MSRC_ObjCategImageDatabase_v2/Images'
OUTPUT_FOLDER = './evaluation_results'
TOP_K = 591  # Evaluate top 10 results
MAX_QUERIES = None  # None = evaluate all queries

# Define evaluation combinations
EVALUATION_COMBINATIONS = [
    {
        'descriptor': 'globalRGBhisto',
        'distance_name': 'euclidean',
        'distance_func': euclidean_distance,
        'pca_model': None,
        'display_name': 'Global RGB Histogram + Euclidean Distance'
    },
    {
        'descriptor': 'globalRGBhisto',
        'distance_name': 'manhattan',
        'distance_func': manhattan_distance,
        'pca_model': None,
        'display_name': 'Global RGB Histogram + Manhattan Distance'
    },
    {
        'descriptor': 'spatialGridRGBhisto',
        'distance_name': 'euclidean',
        'distance_func': euclidean_distance,
        'pca_model': None,
        'display_name': 'Spatial Grid RGB Histogram + Euclidean Distance'
    },
    {
        'descriptor': 'spatialGridRGBhisto',
        'distance_name': 'manhattan',
        'distance_func': manhattan_distance,
        'pca_model': None,
        'display_name': 'Spatial Grid RGB Histogram + Manhattan Distance'
    },
    {
        'descriptor': 'globalRGBhisto_PCA4',
        'distance_name': 'mahalanobis',
        'distance_func': mahalanobis_distance,
        'pca_model': None,  # Will be loaded
        'display_name': 'Global RGB Histogram (PCA 4) + Mahalanobis Distance'
    },
    {
        'descriptor': 'spatialGridRGBhisto_PCA4',
        'distance_name': 'mahalanobis',
        'distance_func': mahalanobis_distance,
        'pca_model': None,  # Will be loaded
        'display_name': 'Spatial Grid RGB Histogram (PCA 4) + Mahalanobis Distance'
    }
]


def save_evaluation_results(results, output_file):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Dictionary containing evaluation results
        output_file: Path to save the JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    results_to_save = {
        'descriptor': results['descriptor'],
        'distance': results['distance'],
        'mean_precision': float(results['mean_precision']),
        'mean_recall': float(results['mean_recall']),
        'mean_ap': float(results['mean_ap']),
        'num_queries': int(results['num_queries']),
        'top_k': TOP_K,
        'all_precisions': [float(p) for p in results['all_precisions']],
        'all_recalls': [float(r) for r in results['all_recalls']],
        'all_aps': [float(ap) for ap in results['all_aps']]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"   💾 Results saved to: {output_file}")


def main():
    """Main evaluation script."""
    print("="*80)
    print("COMPREHENSIVE EVALUATION OF DESCRIPTOR-DISTANCE COMBINATIONS")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"\n📁 Output folder: {OUTPUT_FOLDER}")
    
    # Analyze dataset once
    print("\n" + "="*80)
    print("STEP 1: DATASET ANALYSIS")
    print("="*80)
    dataset_info = analyze_dataset(DATASET_FOLDER)
    
    # Store all evaluation results for comparison
    all_results = []
    
    # Evaluate each combination
    print("\n" + "="*80)
    print("STEP 2: EVALUATING DESCRIPTOR-DISTANCE COMBINATIONS")
    print("="*80)
    
    for idx, combo in enumerate(EVALUATION_COMBINATIONS, 1):
        print(f"\n{'='*80}")
        print(f"EVALUATION {idx}/6: {combo['display_name']}")
        print(f"{'='*80}")
        
        # Load descriptors
        print(f"\n📦 Loading descriptors from: {combo['descriptor']}")
        try:
            all_features, image_paths = load_descriptors(DESCRIPTOR_FOLDER, combo['descriptor'])
            print(f"   ✅ Loaded {len(all_features)} descriptors")
            print(f"   ✅ Descriptor dimension: {all_features.shape[1]}")
        except Exception as e:
            print(f"   ❌ Failed to load descriptors: {e}")
            continue
        
        # Load PCA model if needed
        pca_model = combo['pca_model']
        if combo['descriptor'].endswith('_PCA4'):
            print(f"\n🔧 Loading PCA model for: {combo['descriptor']}")
            pca_model = load_pca_model_for_descriptor(combo['descriptor'])
            if pca_model:
                n_comp = all_features.shape[1]
                var_explained = pca_model['cumulative_variance_ratio'][n_comp - 1] * 100
                print(f"   ✅ PCA model loaded: {n_comp} components ({var_explained:.2f}% variance)")
            else:
                print(f"   ⚠️  Warning: PCA model not found, continuing without it")
        
        # Evaluate all queries
        print(f"\n📊 Evaluating {MAX_QUERIES if MAX_QUERIES else len(all_features)} queries...")
        evaluation_results = evaluate_all_queries(
            all_features=all_features,
            image_paths=image_paths,
            distance_func=combo['distance_func'],
            distance_name=combo['distance_name'],
            pca_model=pca_model,
            top_k=TOP_K,
            max_queries=MAX_QUERIES
        )
        
        # Add metadata
        evaluation_results['descriptor'] = combo['descriptor']
        evaluation_results['distance'] = combo['distance_name']
        evaluation_results['display_name'] = combo['display_name']
        
        # Save PR curve
        pr_curve_filename = f"pr_curve_{combo['descriptor']}_{combo['distance_name']}.png"
        pr_curve_path = os.path.join(OUTPUT_FOLDER, pr_curve_filename)
        
        print(f"\n📈 Plotting PR curve...")
        plot_average_pr_curve(
            evaluation_results['all_pr_curves'],
            title=f"Average PR Curve: {combo['display_name']}",
            save_path=pr_curve_path
        )
        
        # Save evaluation results
        results_filename = f"results_{combo['descriptor']}_{combo['distance_name']}.json"
        results_path = os.path.join(OUTPUT_FOLDER, results_filename)
        
        print(f"\n💾 Saving evaluation results...")
        save_evaluation_results(evaluation_results, results_path)
        
        # Store for summary
        all_results.append(evaluation_results)
    
    # Print summary comparison
    print("\n" + "="*80)
    print("EVALUATION SUMMARY - ALL COMBINATIONS")
    print("="*80)
    
    print(f"\n{'Combination':<50} {'Precision@10':<15} {'Recall@10':<15} {'mAP':<15}")
    print("-" * 95)
    
    for result in all_results:
        print(f"{result['display_name']:<50} "
              f"{result['mean_precision']:>6.4f}        "
              f"{result['mean_recall']:>6.4f}        "
              f"{result['mean_ap']:>6.4f}")
    
    # Save summary comparison
    summary_file = os.path.join(OUTPUT_FOLDER, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION SUMMARY - ALL COMBINATIONS\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Combination':<50} {'Precision@10':<15} {'Recall@10':<15} {'mAP':<15}\n")
        f.write("-" * 95 + "\n")
        
        for result in all_results:
            f.write(f"{result['display_name']:<50} "
                   f"{result['mean_precision']:>6.4f}        "
                   f"{result['mean_recall']:>6.4f}        "
                   f"{result['mean_ap']:>6.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FILES GENERATED:\n")
        f.write("="*80 + "\n\n")
        f.write("PR Curves (PNG files):\n")
        for result in all_results:
            f.write(f"  - pr_curve_{result['descriptor']}_{result['distance']}.png\n")
        f.write("\nEvaluation Results (JSON files):\n")
        for result in all_results:
            f.write(f"  - results_{result['descriptor']}_{result['distance']}.json\n")
    
    print(f"\n📄 Summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ ALL EVALUATIONS COMPLETE!")
    print("="*80)
    print(f"\n📁 All results saved to: {OUTPUT_FOLDER}/")
    print(f"   - 6 PR curve plots (.png)")
    print(f"   - 6 evaluation result files (.json)")
    print(f"   - 1 summary file (.txt)")
    print("="*80)


if __name__ == "__main__":
    main()

