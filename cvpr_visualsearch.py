import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
from cvpr_compare import cvpr_compare

DESCRIPTOR_FOLDER = './descriptors'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'

# Load all descriptors
ALLFEAT = []
ALLFILES = []
IMAGE_FILES = []  # Store actual image file paths

for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
    if filename.endswith('.mat'):
        img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
        img_data = sio.loadmat(img_path)
        ALLFILES.append(img_path)
        ALLFEAT.append(img_data['F'][0])  # Assuming F is a 1D array
        org_image_filename = filename.replace('.mat', '.bmp')
        org_image_path = os.path.join('./MSRC_ObjCategImageDatabase_v2', 'Images', org_image_filename)
        IMAGE_FILES.append(org_image_path)
# Convert ALLFEAT to a numpy array
ALLFEAT = np.array(ALLFEAT)

# Pick a random image as the query
NIMG = ALLFEAT.shape[0]
print("Number of images:", NIMG)
queryimg = randint(0, NIMG - 1)
print("Query image:", queryimg)

# Compute the distance between the query and all other descriptors
dst = []
query = ALLFEAT[queryimg]
for i in range(NIMG):
    candidate = ALLFEAT[i]
    distance = cvpr_compare(query, candidate)
    dst.append((distance, i))

# Sort the distances
dst.sort(key=lambda x: x[0])


# Show the top 15 results
SHOW = 15
# for i in range(SHOW):
#     print("Distance:", dst[i][0], "Image:", ALLFILES[dst[i][1]])
# print(f"Top {SHOW} most similar images:")
# for i in range(SHOW):
#     img_path = IMAGE_FILES[dst[i][1]]
#     # img = cv2.imread(ALLFILES[dst[i][1]])
#     img = cv2.imread(img_path)
#     if img is not None:
#         img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
#         print(f"  {i+1}. {img_path} (distance: {dst[i][0]:.4f})")
#         cv2.imshow(f"Result {i+1}", img)
#         cv2.waitKey(0)
#     else:
#         print(f"  {i+1}. Could not load image: {img_path}")

def load_and_label(path, label_text, target_size=(160, 106)):
    """
    Load an image, resize to target_size, and overlay a label for clarity.
    """
    img = cv2.imread(path)
    if img is None:
        # Create a placeholder image if loading fails
        placeholder = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        cv2.putText(placeholder, 'LOAD FAIL', (10, target_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, label_text, (10, target_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return placeholder
    h, w = img.shape[:2]
    img = cv2.resize(img, target_size)
    # Draw a label background bar for readability
    cv2.rectangle(img, (0, target_size[1] - 18), (target_size[0], target_size[1]), (0, 0, 0), thickness=-1)
    cv2.putText(img, label_text, (6, target_size[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img

# Prepare list: first the query, then the top SHOW results
tiles = []
query_path = IMAGE_FILES[queryimg]
tiles.append(load_and_label(query_path, 'Query'))

for rank in range(SHOW):
    img_index = dst[rank][1]
    img_path = IMAGE_FILES[img_index]
    dist = dst[rank][0]
    tiles.append(load_and_label(img_path, f'{rank+1}: {dist:.3f}'))

# Arrange into a 4x4 grid (16 tiles)
cols = 4
rows = 4
tile_w, tile_h = 160, 106
canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

for idx, tile in enumerate(tiles[:rows * cols]):
    r = idx // cols
    c = idx % cols
    y0, y1 = r * tile_h, (r + 1) * tile_h
    x0, x1 = c * tile_w, (c + 1) * tile_w
    canvas[y0:y1, x0:x1] = tile

print(f"Query image: {query_path}")
print(f"Top {SHOW} most similar images shown in montage window.")
cv2.imshow('Query + Top 15 Results (press any key)', canvas)
cv2.waitKey(0)

cv2.destroyAllWindows()

