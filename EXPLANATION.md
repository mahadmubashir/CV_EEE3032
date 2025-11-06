"""
DETAILED EXPLANATION: Image Processing Pipeline
================================================

This document explains step-by-step how an image becomes a histogram,
and how PCA reduces its dimensions, with concrete examples.
"""

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║          DETAILED EXPLANATION: IMAGE → HISTOGRAM → PCA                  ║
╚══════════════════════════════════════════════════════════════════════════╝

Let's trace through ONE example image from the dataset to understand the flow.

═══════════════════════════════════════════════════════════════════════════
STEP 1: READING THE IMAGE
═══════════════════════════════════════════════════════════════════════════

Example: Image "17_15_s.bmp"

When we read this image using OpenCV:
  img = cv2.imread("17_15_s.bmp")

Result:
  Shape: (213, 320, 3)
  
What does this mean?
  • 213 = Height (number of rows, pixels from top to bottom)
  • 320 = Width (number of columns, pixels from left to right)
  • 3 = Color channels (Blue, Green, Red)
  
Total pixels: 213 × 320 = 68,160 pixels
Each pixel has 3 values: [B, G, R] ranging from 0 to 255

Visual representation:
  ┌─────────────────────────┐
  │ Pixel at (0,0): [B,G,R] │ ← Top-left corner
  │ Pixel at (0,1): [B,G,R] │
  │ Pixel at (0,2): [B,G,R] │
  │         ...             │
  │ Pixel at (212,319): ... │ ← Bottom-right corner
  └─────────────────────────┘
  
  This is a 3D array: rows × columns × channels

═══════════════════════════════════════════════════════════════════════════
STEP 2: NORMALIZING THE IMAGE
═══════════════════════════════════════════════════════════════════════════

We convert pixel values from [0, 255] to [0, 1]:
  img_normalized = img.astype(np.float64) / 255.0

Result:
  Shape: (213, 320, 3)  ← Same shape!
  Values: Now between 0.0 and 1.0

Why normalize?
  • Makes calculations easier (no large numbers)
  • Ensures consistency across different image formats
  • Required for histogram computation

═══════════════════════════════════════════════════════════════════════════
STEP 3: COMPUTING GLOBAL RGB HISTOGRAM
═══════════════════════════════════════════════════════════════════════════

What is a histogram?
  A histogram counts how many pixels fall into different "bins" (ranges).

Example with 8 bins:
  Bin 0: [0-31]    (darkest colors)
  Bin 1: [32-63]
  Bin 2: [64-95]
  Bin 3: [96-127]
  Bin 4: [128-159]
  Bin 5: [160-191]
  Bin 6: [192-223]
  Bin 7: [224-255] (brightest colors)

Process:
  1. Separate the image into Red, Green, Blue channels
  2. For EACH channel, count pixels in each bin
  3. Normalize each histogram (divide by total pixels) so it sums to 1
  4. Concatenate all three histograms together

Example computation:
  Red channel:   [0.069, 0.071, 0.101, 0.168, 0.236, 0.122, 0.059, 0.174]
                 ↑ bin0  ↑ bin1  ↑ bin2  ↑ bin3  ↑ bin4  ↑ bin5  ↑ bin6  ↑ bin7
                 (6.9% of pixels in bin 0, 7.1% in bin 1, etc.)
  
  Green channel: [0.063, 0.073, 0.103, 0.183, 0.238, 0.112, 0.049, 0.177]
  
  Blue channel:  [0.077, 0.083, 0.118, 0.192, 0.238, 0.075, 0.035, 0.182]

Final histogram descriptor:
  [R₀, R₁, R₂, R₃, R₄, R₅, R₆, R₇, G₀, G₁, G₂, G₃, G₄, G₅, G₆, G₇, B₀, B₁, B₂, B₃, B₄, B₅, B₆, B₇]
  
  Shape: (1, 24)  ← Single row, 24 columns
         8 bins × 3 channels = 24 total values

What does this represent?
  • Instead of 68,160 pixels (213×320×3), we have 24 numbers
  • These 24 numbers summarize the COLOR DISTRIBUTION of the image
  • Images with similar colors will have similar histograms

═══════════════════════════════════════════════════════════════════════════
STEP 4: COLLECTING DESCRIPTORS FROM ALL IMAGES
═══════════════════════════════════════════════════════════════════════════

We repeat Step 3 for ALL 591 images in the dataset.

Result: A 2D matrix where:
  • Each ROW = one image's histogram descriptor
  • Each COLUMN = one histogram bin
  
Shape: (591, 24)

Visual representation:
  ┌─────────────────────────────────────────┐
  │ Image 1: [R₀ R₁ ... R₇ G₀ ... B₇]      │ ← Row 0
  │ Image 2: [R₀ R₁ ... R₇ G₀ ... B₇]      │ ← Row 1
  │ Image 3: [R₀ R₁ ... R₇ G₀ ... B₇]      │ ← Row 2
  │   ...                                   │
  │ Image 591: [R₀ R₁ ... R₇ G₀ ... B₇]   │ ← Row 590
  └─────────────────────────────────────────┘
  
  This is a 2D array: (number of images) × (descriptor dimension)

═══════════════════════════════════════════════════════════════════════════
STEP 5: COMPUTING PCA (PRINCIPAL COMPONENT ANALYSIS)
═══════════════════════════════════════════════════════════════════════════

What is PCA?
  PCA finds the "most important directions" in your data.
  
Think of it like this:
  • Imagine all 591 images as points in a 24-dimensional space
  • PCA finds the directions where these points spread out the MOST
  • The first principal component = direction of maximum spread
  • The second principal component = direction of second-most spread (perpendicular to first)
  • And so on...

Mathematical process:
  1. Compute the covariance matrix: (24, 24)
     - Shows how different histogram bins vary together
     - Diagonal = variance of each bin
     - Off-diagonal = covariance between bins
  
  2. Find eigenvectors (principal components):
     - These are the directions of maximum variance
     - Shape: (24, 24) - Each column is one principal component
     - First column = most important direction
     - Second column = second most important direction
     - etc.
  
  3. Compute explained variance:
     - How much of the total variance each component captures
     - Example: Component 1 might capture 40% of variance
     - Component 2 might capture 30% of variance
     - etc.

Result: PCA Model
  • Components matrix: (24, 24)
    - Each column is a principal component (direction vector)
    - Each component is a 24-dimensional vector
  
  • Explained variance: How important each component is
    - Component 1: 40% variance
    - Component 2: 30% variance
    - Component 3: 15% variance
    - Component 4: 10% variance
    - Components 5-24: 5% variance total

Key insight:
  The first few components capture MOST of the information!
  Components 1-10 might capture 90% of the variance.

═══════════════════════════════════════════════════════════════════════════
STEP 6: PROJECTING TO LOWER DIMENSIONS (PCA 10)
═══════════════════════════════════════════════════════════════════════════

Instead of using all 24 dimensions, we use only the top 10 components.

Process:
  For each image's 24-dimensional descriptor:
    1. Project it onto the first 10 principal components
    2. This gives us 10 numbers instead of 24
    3. These 10 numbers represent the image in the "PCA space"

Mathematical operation:
  Original descriptor: [24 numbers]
  Projection: Multiply by first 10 principal components
  Result: [10 numbers]

Example:
  Image 1 original: [R₀, R₁, ..., R₇, G₀, ..., B₇]  (24 numbers)
  Image 1 PCA 10:    [PC₁, PC₂, PC₃, ..., PC₁₀]      (10 numbers)

Result:
  Shape: (591, 10)  ← Reduced from (591, 24)
  
  Each image now represented by 10 numbers instead of 24!
  
  Variance preserved: ~89.5% (we keep most of the information!)

Visual representation:
  ┌─────────────────────────┐
  │ Image 1: [PC₁ ... PC₁₀]│ ← Row 0 (10 numbers)
  │ Image 2: [PC₁ ... PC₁₀]│ ← Row 1 (10 numbers)
  │ Image 3: [PC₁ ... PC₁₀]│ ← Row 2 (10 numbers)
  │   ...                   │
  │ Image 591: [PC₁ ... PC₁₀]│ ← Row 590 (10 numbers)
  └─────────────────────────┘

Benefits:
  • Storage: 58% less (10 vs 24 numbers per image)
  • Speed: Faster distance computations
  • Noise reduction: Less important dimensions removed
  • Information: Still keeps 89.5% of variance!

═══════════════════════════════════════════════════════════════════════════
COMPLETE PIPELINE SUMMARY
═══════════════════════════════════════════════════════════════════════════

Image File (17_15_s.bmp)
    │
    ▼
(213, 320, 3) - 68,160 pixels, values [0-255]
    │
    ▼ Normalize: ÷ 255
(213, 320, 3) - Values [0.0-1.0]
    │
    ▼ Compute histogram (8 bins × 3 channels)
(1, 24) - Single descriptor vector
    │
    ▼ Repeat for all 591 images
(591, 24) - All image descriptors
    │
    ▼ Compute PCA
PCA Model:
  - Components: (24, 24)
  - Explained variance: [40%, 30%, 15%, ...]
    │
    ▼ Project onto top 10 components
(591, 10) - Reduced dimension descriptors
    │
    ▼ Use for similarity search
Compare images using only 10 numbers instead of 24!

═══════════════════════════════════════════════════════════════════════════
KEY TAKEAWAYS
═══════════════════════════════════════════════════════════════════════════

1. IMAGE → HISTOGRAM
   • Image: 68,160 numbers (pixels)
   • Histogram: 24 numbers (color distribution summary)
   • Compression: 99.96% reduction!

2. HISTOGRAM → PCA
   • Original: 24 dimensions
   • PCA 10: 10 dimensions
   • Compression: 58% reduction
   • Information preserved: 89.5%

3. WHY PCA WORKS
   • Not all dimensions are equally important
   • Some dimensions are correlated (redundant)
   • PCA finds the most informative directions
   • We keep only the important ones

4. ANALOGY
   • Think of describing a person:
     - Full description: height, weight, age, hair color, eye color, 
       shoe size, shirt color, pants color, etc. (many dimensions)
     - PCA: Find the most distinguishing features
     - Reduced description: "tall, dark hair, blue eyes" (key features)
   • Same idea with images: Keep the most distinguishing features!

═══════════════════════════════════════════════════════════════════════════
""")

