# import os
# import numpy as np
# import scipy.io as sio
# import cv2
# from random import randint
# from cvpr_compare import cvpr_compare
# import matplotlib.pyplot as plt
# DESCRIPTOR_FOLDER = './descriptors'
# DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
# EVALUATE = True  # Set True to compute Precision-Recall metrics
# TOP_K = 10       # Number of top results to visualise
# SAVE_FIGS = True # Save PR curve and confusion matrix plots
# MATPLOTLIB_AVAILABLE = True # Optional plotting (gracefully degrade if matplotlib not available)

# # Load all descriptors
# ALLFEAT = []
# ALLFILES = []
# IMAGE_FILES = []  # Store actual image file paths

# for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
#     if filename.endswith('.mat'):
#         img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
#         img_data = sio.loadmat(img_path)
#         ALLFILES.append(img_path)
#         ALLFEAT.append(img_data['F'][0])  # Assuming F is a 1D array
#         org_image_filename = filename.replace('.mat', '.bmp')
#         org_image_path = os.path.join('./MSRC_ObjCategImageDatabase_v2', 'Images', org_image_filename)
#         IMAGE_FILES.append(org_image_path)
# # Convert ALLFEAT to a numpy array
# ALLFEAT = np.array(ALLFEAT)

# # Pick a random image as the query
# NIMG = ALLFEAT.shape[0]
# print("Number of images:", NIMG)
# queryimg = randint(0, NIMG - 1)
# print("Query image:", queryimg)

# # Compute the distance between the query and all other descriptors
# dst = []
# query = ALLFEAT[queryimg]
# for i in range(NIMG):
#     candidate = ALLFEAT[i]
#     distance = cvpr_compare(query, candidate)
#     dst.append((distance, i))

# # Sort the distances (ascending by distance)
# dst.sort(key=lambda x: x[0])


# # Show the top 15 results
# # SHOW = 10
# SHOW = TOP_K
# # for i in range(SHOW):
# #     print("Distance:", dst[i][0], "Image:", ALLFILES[dst[i][1]])
# # print(f"Top {SHOW} most similar images:")
# # for i in range(SHOW):
# #     img_path = IMAGE_FILES[dst[i][1]]
# #     # img = cv2.imread(ALLFILES[dst[i][1]])
# #     img = cv2.imread(img_path)
# #     if img is not None:
# #         img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
# #         print(f"  {i+1}. {img_path} (distance: {dst[i][0]:.4f})")
# #         cv2.imshow(f"Result {i+1}", img)
# #         cv2.waitKey(0)
# #     else:
# #         print(f"  {i+1}. Could not load image: {img_path}")

# def load_and_label(path, label_text, target_size=(160, 106)):
#     """
#     Load an image, resize to target_size, and overlay a label for clarity.
#     """
#     img = cv2.imread(path)
#     if img is None:
#         # Create a placeholder image if loading fails
#         placeholder = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
#         cv2.putText(placeholder, 'LOAD FAIL', (10, target_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#         cv2.putText(placeholder, label_text, (10, target_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
#         return placeholder
#     h, w = img.shape[:2]
#     img = cv2.resize(img, target_size)
#     # Draw a label background bar for readability
#     cv2.rectangle(img, (0, target_size[1] - 18), (target_size[0], target_size[1]), (0, 0, 0), thickness=-1)
#     cv2.putText(img, label_text, (6, target_size[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
#     return img

# # Enlarge tiles for better visibility
# LARGE_TILE_W, LARGE_TILE_H = 320, 212  # Doubled width and height

# # Prepare list: first the query, then the top SHOW results
# tiles = []
# query_path = IMAGE_FILES[queryimg]
# tiles.append(load_and_label(query_path, 'Query', target_size=(LARGE_TILE_W, LARGE_TILE_H)))

# for rank in range(SHOW):
#     img_index = dst[rank][1]
#     img_path = IMAGE_FILES[img_index]
#     dist = dst[rank][0]
#     tiles.append(load_and_label(img_path, f'{rank+1}: {dist:.3f}', target_size=(LARGE_TILE_W, LARGE_TILE_H)))

# # (Montage grid: 1 row for query, 1 row for top 10 results -- now with larger tiles)
# cols = 10
# rows = 2
# tile_w, tile_h = LARGE_TILE_W, LARGE_TILE_H
# canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

# # First row: show the query image in the center, rest blank
# for c in range(cols):
#     if c == cols // 2:  # Place query at center position
#         tile = tiles[0]
#         y0, y1 = 0, tile_h
#         x0, x1 = c * tile_w, (c + 1) * tile_w
#         canvas[y0:y1, x0:x1] = tile
#     else:
#         # leave blank (already zeroed)
#         continue

# # Second row: top 10 results (tiles[1:11])
# for i, tile in enumerate(tiles[1:11]):
#     y0, y1 = tile_h, 2 * tile_h
#     x0, x1 = i * tile_w, (i + 1) * tile_w
#     canvas[y0:y1, x0:x1] = tile

# print(f"Query image: {query_path}")
# print(f"Top {SHOW} most similar images shown in montage window.")
# cv2.imshow('Query + Top Results (press any key)', canvas)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# def get_class_id_from_path(path):
#     """
#     Extract the integer class id from an image path like './.../12_33_s.bmp' -> 12
#     """
#     base = os.path.basename(path)
#     try:
#         return int(base.split('_')[0])
#     except Exception:
#         return -1
        
# # Print class_ids for top 10 ranked images only (excluding the query itself)
# print(f"\nTop {TOP_K} ranked results with class IDs:")
# ranked_all = [(d, i) for (d, i) in dst if i != queryimg]
# # Limit to top 10 only
# for i in range(TOP_K):
#     d, idx = ranked_all[i]
#     rank = i + 1
#     cls_id = get_class_id_from_path(IMAGE_FILES[idx])
#     print(f"{rank:4d}: class={cls_id:2d}, dist={d:.4f}, file={os.path.basename(IMAGE_FILES[idx])}")

# # =====================
# # Precision-Recall Eval
# # =====================

# def compute_precision_recall(sorted_indices, query_class_id, k=None):
#     """
#     Compute precision and recall arrays for a ranked list of indices.
    
#     Parameters
#     ----------
#     - sorted_indices: list[int]
#         Dataset indices ranked by increasing distance to the query (best first).
#         Typically produced from 'dst' after sorting; may or may not include the
#         query itself (callers often exclude self before passing here).
#     - query_class_id: int
#         Ground-truth class for the query image (e.g., parsed as the first token
#         of the filename '12_33_s.bmp' -> 12).
#     - k: Optional[int]
#         Evaluate only the first k ranked items (defaults to full list).
    
#     Returns
#     -------
#     - precisions: list[float]
#         Precision at each rank position r (1..k): TP(r) / r
#     - recalls: list[float]
#         Recall at each rank position r (1..k): TP(r) / TP_total
#     - average_precision: float
#         AP computed as mean of precisions at ranks where a relevant item is found.
#     - precision_at_k: float
#         Precision measured at rank k (or at the end if less than k results).
#     - recall_at_k: float
#         Recall measured at rank k (or at the end if less than k results).
#     """
#     if k is None:
#         k = len(sorted_indices)
#     # Total number of relevant images in the dataset for the query's class.
#     # Note: if the ranked list still contains the query itself, that will be counted
#     # here as relevant_total as well — which is fine, because callers typically pass
#     # a list with self excluded when computing PR for retrieval.
#     relevant_total = sum(
#         1 for idx in range(len(IMAGE_FILES))
#         if get_class_id_from_path(IMAGE_FILES[idx]) == query_class_id
#     )
#     print(f"Relevant total: {relevant_total}, Query class id: {query_class_id}")
#     if relevant_total == 0:
#         return [], [], 0.0, 0.0, 0.0

#     # Running counters and outputs
#     precisions = []
#     recalls = []
#     num_relevant_found = 0

#     # Average Precision (AP) accumulator. Each time we hit a relevant item,
#     # we add the precision at that rank; AP = sum(precision_at_relevant_ranks)/relevant_total
#     ap_sum = 0.0

#     # Iterate through the ranked list up to k
#     for rank, idx in enumerate(sorted_indices[:k], start=1):
#         # Determine if the current ranked item is relevant
#         is_relevant = (get_class_id_from_path(IMAGE_FILES[idx]) == query_class_id)
#         # print(f"Is relevant: {is_relevant}, Query class id: {query_class_id}, Image file: {IMAGE_FILES[idx]}")
#         # Update true positives if relevant
#         if is_relevant:
#             num_relevant_found += 1
#             # Accumulate precision at this rank for AP computation
#             ap_sum += num_relevant_found / rank
#             # print(f"AP sum: {ap_sum}")
#         # Precision = TP(r) / r  (how many relevant among the top r retrieved)
#         precisions.append(num_relevant_found / rank)
#         # print(f"Precision at rank {rank}: {num_relevant_found / rank}")
#         # Recall = TP(r) / TP_total (how many relevant found vs how many exist overall)
#         recalls.append(num_relevant_found / relevant_total)
#         # print(f"Recall at rank {rank}: {num_relevant_found / relevant_total}")
#     print(f"Precisions: {precisions}")
#     print(f"Recalls: {recalls}")
#     average_precision = ap_sum / relevant_total if relevant_total > 0 else 0.0
#     print(f"Average precision: {average_precision}")
#     precision_at_k = precisions[-1] if precisions else 0.0
#     print(f"Precision at k: {precision_at_k}")
#     recall_at_k = recalls[-1] if recalls else 0.0
#     print(f"Recall at k: {recall_at_k}")
#     return precisions, recalls, average_precision, precision_at_k, recall_at_k

# if EVALUATE:
#     # ---------- Single-query metrics ----------
#     # Exclude the query itself from the ranked list, then evaluate only top 10
#     ranked_indices = [idx for (_, idx) in dst if idx != queryimg]
#     query_class = get_class_id_from_path(IMAGE_FILES[queryimg])
#     # Evaluate only top 10 results as per requirement
#     P, R, AP, P_at_K, R_at_K = compute_precision_recall(ranked_indices, query_class, k=TOP_K)
#     print("\nEvaluation (single query):")
#     print(f"  Query class: {query_class}")
#     print(f"  Average Precision (AP): {AP:.4f}")
#     print(f"  Precision@{TOP_K}: {P_at_K:.4f}")
#     print(f"  Recall@{TOP_K}: {R_at_K:.4f}")

#     # ---------- Full-dataset evaluation ----------
#     print("\nEvaluating all queries (this may take a little time)...")

#     # Precompute class ids for all images
#     class_ids = [get_class_id_from_path(p) for p in IMAGE_FILES]
#     num_classes = max(class_ids) if class_ids else 0
#     print(f"Number of classes: {num_classes}")
#     confusion = np.zeros((num_classes, num_classes), dtype=np.int32)

#     ap_list = []
#     p_at_k_list = []
#     r_at_k_list = []

#     # For mean PR curve: accumulate interpolated precision at 11 recall levels
#     recall_levels = [i / 10 for i in range(11)]
#     interp_precisions_accum = np.zeros(len(recall_levels), dtype=np.float64)

#     for qidx in range(NIMG):
#         q_feat = ALLFEAT[qidx]
#         # Compute distances for this query
#         dists = []
#         for i in range(NIMG):
#             d = cvpr_compare(q_feat, ALLFEAT[i])
#             dists.append((d, i))
#         dists.sort(key=lambda x: x[0])
#         ranked = [i for (_, i) in dists if i != qidx]  # exclude self

#         q_class = class_ids[qidx]
#         # Evaluate only top 10 results as per requirement
#         Pq, Rq, APq, Pkq, Rkq = compute_precision_recall(ranked, q_class, k=TOP_K)
#         ap_list.append(APq)
#         p_at_k_list.append(Pkq if len(ranked) >= TOP_K else 0.0)
#         r_at_k_list.append(Rkq if len(ranked) >= TOP_K else 0.0)

#         # For PR curve: use only the top 10 results
#         # Build PR curve from the 10 precision-recall pairs (one per rank)
#         if len(Pq) > 0 and len(Rq) > 0:
#             # For each recall level, find max precision from top 10 results
#             for ri, r_level in enumerate(recall_levels):
#                 max_p = 0.0
#                 for p, r in zip(Pq, Rq):
#                     if r >= r_level and p > max_p:
#                         max_p = p
#                 interp_precisions_accum[ri] += max_p

#         # Confusion matrix using top-1 retrieved class (excluding self)
#         if ranked:
#             top1 = ranked[0]
#             pred_class = class_ids[top1]
#             if 1 <= q_class <= num_classes and 1 <= pred_class <= num_classes:
#                 confusion[q_class - 1, pred_class - 1] += 1

#     mAP = float(np.mean(ap_list)) if ap_list else 0.0
#     mean_P_at_K = float(np.mean(p_at_k_list)) if p_at_k_list else 0.0
#     mean_R_at_K = float(np.mean(r_at_k_list)) if r_at_k_list else 0.0

#     mean_interp_precisions = interp_precisions_accum / max(NIMG, 1)

#     print("\nEvaluation (all queries):")
#     print(f"  mAP: {mAP:.4f}")
#     print(f"  mean Precision@{TOP_K}: {mean_P_at_K:.4f}")
#     print(f"  mean Recall@{TOP_K}: {mean_R_at_K:.4f}")

#     # Save/plot PR curve (based on top 10 results only)
#     if MATPLOTLIB_AVAILABLE and SAVE_FIGS:
#         plt.figure(figsize=(5,4))
#         plt.plot(recall_levels, mean_interp_precisions, marker='o', label='Interpolated (11-point)')
#         plt.title(f'Mean PR Curve (Top {TOP_K} Results)')
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig('pr_curve.png', dpi=150)
#         plt.close()
#         print(f"  Saved PR curve (top {TOP_K} results) to pr_curve.png")
#         # Save CSV if plotting not available
#         out_csv = 'pr_curve.csv'
#         with open(out_csv, 'w') as f:
#             f.write('recall,precision\n')
#             for r, p in zip(recall_levels, mean_interp_precisions):
#                 f.write(f"{r},{p}\n")
#         print("  Saved PR points to pr_curve.csv")

#     # Plot/save confusion matrix
#     if MATPLOTLIB_AVAILABLE and SAVE_FIGS and num_classes > 0:
#         import matplotlib.ticker as mticker
#         plt.figure(figsize=(6,5))
#         # Normalize rows to percentages
#         row_sums = confusion.sum(axis=1, keepdims=True) + 1e-12
#         conf_norm = confusion / row_sums
#         plt.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
#         plt.title('Confusion Matrix (Top-1 Retrieval)')
#         plt.xlabel('Predicted Class')
#         plt.ylabel('True Class')
#         plt.colorbar(format=mticker.FormatStrFormatter('%.2f'))
#         ticks = np.arange(0, num_classes)
#         plt.xticks(ticks, [str(i+1) for i in ticks], rotation=90)
#         plt.yticks(ticks, [str(i+1) for i in ticks])
#         plt.tight_layout()
#         plt.savefig('confusion_matrix.png', dpi=150)
#         plt.close()
#         print("  Saved confusion matrix to confusion_matrix.png")
#     else:
#         # Print small confusion summary (diagonal accuracy)
#         correct = int(np.trace(confusion))
#         total = int(np.sum(confusion))
#         acc = correct / total if total > 0 else 0.0
#         print(f"  Top-1 retrieval accuracy (diagonal of confusion): {acc:.4f}")

