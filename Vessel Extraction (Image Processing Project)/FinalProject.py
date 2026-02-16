import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import glob
import os
from sklearn.metrics import confusion_matrix
from PIL import Image


def get_fov_bbox(mask):
    """
    Finds the four extreme coordinates (bounding box) of the FOV mask.
    Corresponds to paper Section 3.2.
    
    Args:
        mask (numpy array): Binary FOV mask (0 is background, 255 is ROI).
        
    Returns:
        tuple: (y_min, y_max, x_min, x_max) coordinates for cropping.
    """
    # Find indices where the mask is white (255)
    # This effectively finds the "four extreme coordinates" mentioned in the paper
    points = cv2.findNonZero(mask)
    
    if points is None:
        return 0, mask.shape[0], 0, mask.shape[1]

    # Get the bounding rectangle of the non-zero points
    x, y, w, h = cv2.boundingRect(points)
    
    # Calculate extreme coordinates
    x_min = x
    x_max = x + w
    y_min = y
    y_max = y + h
    
    return y_min, y_max, x_min, x_max

def crop_to_fov(img, bbox):
    """
    Crops the image using the bounding box coordinates.
    Reproduces Fig. 2(c) from the paper.
    """
    y_min, y_max, x_min, x_max = bbox
    return img[y_min:y_max, x_min:x_max]

def extract_channels(img_rgb):
    """
    Transforms cropped image into L*a*b and YCbCr spaces and selects
    G, L, and Y channels.
    Corresponds to paper Section 3.2.
    
    Args:
        img_rgb (numpy array): Cropped RGB image.
        
    Returns:
        dict: Dictionary containing 'G', 'L', and 'Y' channels.
    """
    # 1. Select Green (G) from RGB
    # OpenCV loads images as BGR by default, so Green is at index 1
    # If passing RGB, Green is still index 1.
    g_channel = img_rgb[:, :, 1]

    # 2. Convert to CIE L*a*b* and select Luminance (L)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    l_channel = img_lab[:, :, 0] # L is the 0th channel

    # 3. Convert to YCbCr and select Luma (Y)
    img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y_channel = img_ycbcr[:, :, 0] # Y is the 0th channel
    
    return {'G': g_channel, 'L': l_channel, 'Y': y_channel}

def apply_clahe(channels):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the channels.
    Corresponds to paper Section 3.2.
    
    Parameters per paper:
    - Tile Size: 8x8
    - Clip Limit: 0.01 (MATLAB scale) -> ~2.0 in OpenCV scale
    """
    # Create CLAHE object
    # tileGridSize is explicitly set to (8,8) as per paper 
    # clipLimit is set to 2.0 (standard OpenCV equivalent for MATLAB's 0.01)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    enhanced_channels = {}
    for name, img in channels.items():
        enhanced_channels[name] = clahe.apply(img)
        
    return enhanced_channels

def preprocess_pipeline(image_path, mask_path):
    """
    Executes the full preprocessing pipeline described in Fig. 1 of the paper.
    """
    # Load images
    # Note: OpenCV loads as BGR. We convert to RGB for consistency with paper terms.
    bgr_img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if bgr_img is None or mask is None:
        raise FileNotFoundError("Could not find image or mask.")
        
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # --- Step 1: Cropping (Paper Fig. 2) ---
    print("Step 1: Finding FOV coordinates...")
    bbox = get_fov_bbox(mask)
    rgb_cropped = crop_to_fov(rgb_img, bbox)
    mask_cropped = crop_to_fov(mask, bbox) # We usually crop the mask too for later steps
    
    # --- Step 2: Channel Selection (Paper Fig. 3) ---
    print("Step 2: Converting color spaces and selecting G, L, Y channels...")
    channels = extract_channels(rgb_cropped)
    
    # --- Step 3: Contrast Enhancement (Paper Fig. 4) ---
    print("Step 3: Applying Adaptive Contrast Enhancement (CLAHE)...")
    enhanced_channels = apply_clahe(channels)
    
    return rgb_cropped, channels, enhanced_channels

# # --- Example Usage Code ---
# # To run this, you need a sample 'image.tif' and 'mask.gif' from the DRIVE dataset.
# if __name__ == "__main__":
#     # Replace these with actual paths from the DRIVE dataset
#     # You can download DRIVE here: https://drive.grand-challenge.org/
#     img_path = 'DRIVE/test/images/01_test.tif' 
#     mask_path = 'DRIVE/test/mask/01_test_mask.gif'
    
#     try:
#         cropped, original_channels, final_channels = preprocess_pipeline(img_path, mask_path)

#         # Visualization to verify against Paper Figures
#         plt.figure(figsize=(10, 8))

#         # Compare G channel: Raw vs Enhanced (Matches Fig 4a vs Fig 3a)
#         plt.subplot(2, 3, 1)
#         plt.imshow(original_channels['G'], cmap='gray')
#         plt.title("Original G (Fig 3a)")
        
#         plt.subplot(2, 3, 4)
#         plt.imshow(final_channels['G'], cmap='gray')
#         plt.title("Enhanced G (Fig 4a)")

#         # Compare Y channel
#         plt.subplot(2, 3, 2)
#         plt.imshow(original_channels['Y'], cmap='gray')
#         plt.title("Original Y (Fig 3b)")
        
#         plt.subplot(2, 3, 5)
#         plt.imshow(final_channels['Y'], cmap='gray')
#         plt.title("Enhanced Y (Fig 4b)")

#         # Compare L channel
#         plt.subplot(2, 3, 3)
#         plt.imshow(original_channels['L'], cmap='gray')
#         plt.title("Original L (Fig 3c)")
        
#         plt.subplot(2, 3, 6)
#         plt.imshow(final_channels['L'], cmap='gray')
#         plt.title("Enhanced L (Fig 4c)")

#         plt.tight_layout()
#         plt.show()
        
#     except Exception as e:
#         print(f"Error: {e}")
#         print("Please ensure you have the DRIVE dataset images to run the example.")

# --- Helper Function for Gabor Sigma ---
def get_sigma(lambd, bandwidth=1.0):
    """
    Calculates Sigma based on Wavelength and Bandwidth.
    Derived from standard Gabor filter formulas.
    Ref: Paper mentions bandwidth = 1.
    """
    # Formula: sigma = (lambda / pi) * sqrt(ln(2)/2) * ((2^b + 1) / (2^b - 1))
    # For b = 1, the last term is 3.
    # sigma approx 0.56 * lambda
    return (lambd / np.pi) * np.sqrt(np.log(2) / 2) * ((2**bandwidth + 1) / (2**bandwidth - 1))

#--- 1. Multiscale Gabor Filtering ---
def apply_multiscale_gabor(channels):
    """
    Extracts Gabor features at 3 wavelengths from (G, Y, L) channels.
    
    Paper Specifications:
    - Wavelengths: 9, 10, 11 
    - Aspect Ratio (gamma): 0.5 
    - Bandwidth: 1 
    - Orientations: "starts from 0 and occurs at every 15 angle".
      (360 / 15 = 24 orientations).
    - "bank of 72 Gabor filter is created"  (24 orientations * 3 wavelengths = 72).
    - "9 images are produced in this step"[cite: 237].
      (Max response across orientations for each of 3 wavelengths on 3 channels).
      
    Args:
        channels (dict): Dictionary {'G': img, 'Y': img, 'L': img}
        
    Returns:
        dict: Keys are strings like 'G_9', 'Y_11', etc. Values are the filtered images.
    """
    wavelengths = [9, 10, 11]
    # 0 to 360 step 15 gives 24 orientations. 24 * 3 wavelengths = 72 filters.
    angles = np.arange(0, 360, 15) 
    gamma = 0.5
    psi = 0 
    
    gabor_features = {}
    
    print(f"Extracting Gabor features for wavelengths {wavelengths}...")
    
    # Iterate over all 3 channels (G, Y, L) as required by [cite: 237]
    for ch_name, img in channels.items():
        # Important: Vessels are dark in fundus images. Gabor filters with psi=0
        # respond to bright lines. We invert the image so vessels become bright structures.
        # This fixes the "hollow vessel" or "empty filled" issue.
        img_inverted = cv2.bitwise_not(img)
        
        for lambd in wavelengths:
            sigma = get_sigma(lambd, bandwidth=1.0)
            
            # Accumulator for max response across the 24 orientations
            max_response = np.zeros_like(img, dtype=np.float32)
            
            for theta_deg in angles:
                theta_rad = np.deg2rad(theta_deg)
                
                # Create Kernel
                # ksize must be large enough to contain the vessel response.
                # 6*sigma is standard, but ensuring it's odd.
                ksize = int(6 * sigma) 
                if ksize % 2 == 0: ksize += 1
                
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta_rad, lambd, gamma, psi, ktype=cv2.CV_32F)
                
                # Apply Filter
                filtered = cv2.filter2D(img_inverted, cv2.CV_32F, kernel)
                
                # Take element-wise maximum across orientations [cite: 227]
                # "scan whole the image in all orientations and extract all the vessels"
                max_response = np.maximum(max_response, filtered)
                
            # Normalize to 0-255
            max_response = cv2.normalize(max_response, None, 0, 255, cv2.NORM_MINMAX)
            gabor_features[f"{ch_name}_{lambd}"] = max_response.astype(np.uint8)

    return gabor_features

def automatic_thresholding(gabor_images):
    """
    Performs binarization on the 9 Gabor images using Laplace filter and calculated threshold.
    
    MODIFIED: Calculates threshold based ONLY on non-zero pixels (inside FOV) 
    to prevent black borders from skewing the mean intensity downwards.
    """
    binary_images = {}
    
    print("Applying Automatic Thresholding on 9 Gabor images...")
    
    for key, img in gabor_images.items():
        # 1. Apply Laplace Filter [cite: 240] (with Gaussian Blur for noise [cite: 61])
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        laplace = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
        laplace = cv2.convertScaleAbs(laplace)
        
        # 2. Add Laplace image to Gabor image [cite: 241]
        combined = cv2.add(img, laplace)
        
        # 3. Calculate Threshold [cite: 242-248]
        # FIX: Create a mask for non-black pixels to exclude the FOV border
        # This ensures the mean is calculated only for the actual retinal area.
        mask = (combined > 0).astype(np.uint8)
        
        # Calculate mean intensity only where mask is > 0
        # cv2.mean returns a tuple (mean_val, 0, 0, 0), we take the first element.
        mean_intensity = cv2.mean(combined, mask=mask)[0]
        
        # The paper defines the threshold as this weighted sum (Mean Intensity)
        threshold_val = int(mean_intensity)
        
        # 4. Apply Threshold
        _, binary = cv2.threshold(combined, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Ensure the binary mask is also clipped to FOV (remove any border artifacts)
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        
        binary_images[key] = binary
        
    return binary_images

# --- 3. Morphological & Image Processing Features (TH, SC, BPS) ---
def extract_special_features(g_channel):
    """
    Extracts Top Hat (TH), Shade Corrected (SC), and Bit Plane Slicing (BPS) features.
    
    Updated Specifications:
    - TH: Extracted from inverted G channel.
    - SC: Extracted from G channel.
    - BPS: Extracted from the Top-Hat (TH) result.
           *Resize modification*: Image is resized to (512, 512) for BPS calculation
           to reduce high-frequency bit noise, then resized back.
    """
    features = {}
    print("Extracting TH, SC, and BPS features from G channel...")

    # --- Feature: Top Hat (TH) ---
    g_inverted = cv2.bitwise_not(g_channel)
    th_max = np.zeros_like(g_inverted)
    line_len = 21
    angles = np.arange(0, 180, 22.5)
    
    for angle in angles:
        kernel_size = line_len
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2
        angle_rad = np.deg2rad(angle)
        x_offset = int((line_len/2) * np.cos(angle_rad))
        y_offset = int((line_len/2) * np.sin(angle_rad))
        p1 = (center - x_offset, center - y_offset)
        p2 = (center + x_offset, center + y_offset)
        cv2.line(kernel, p1, p2, 1, thickness=1)
        tophat = cv2.morphologyEx(g_inverted, cv2.MORPH_TOPHAT, kernel)
        th_max = np.maximum(th_max, tophat)
        
    features['TH'] = th_max

    # --- Feature: Shade Corrected (SC) ---
    bg = cv2.medianBlur(g_channel, 25)
    features['SC'] = cv2.absdiff(g_channel, bg)

    # --- Feature: Bit Plane Slicing (BPS) ---
    # Logic: Use TH output -> Resize -> Extract Planes -> Sum -> Resize Back
    
    # 1. Store original size
    original_h, original_w = th_max.shape
    
    # 2. Resize for BPS processing
    # Using 512x512 as a stable standard size to reduce noise in bit planes
    target_size = (512, 512)
    bps_source_resized = cv2.resize(th_max, target_size, interpolation=cv2.INTER_AREA)
    
    bps_planes_viz = []
    
    # 3. Extract planes from RESIZED image
    for i in range(8):
        # Shift and mask to get the specific bit (0 or 1)
        plane = ((bps_source_resized >> i) & 1) * 255
        
        # Resize back to ORIGINAL size immediately for visualization storage
        # Use INTER_NEAREST to keep binary sharp edges (0 or 255 only)
        plane_restored = cv2.resize(plane.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        bps_planes_viz.append(plane_restored)
        
    features['BPS_planes'] = bps_planes_viz

    # 4. Calculate Feature Vector from RESIZED image (Planes 7 and 8)
    plane_8 = (bps_source_resized >> 7) & 1
    plane_7 = (bps_source_resized >> 6) & 1
    
    bps_sum_resized = plane_8 + plane_7
    
    # Scale to 0-255 range (0, 127, 254)
    bps_sum_scaled = (bps_sum_resized * 127).astype(np.uint8)
    
    # 5. Resize Result Back to Original Size
    # Use INTER_NEAREST to preserve the specific levels (0, 127, 254) without blurring
    features['BPS'] = cv2.resize(bps_sum_scaled, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    return features

def construct_feature_vector(binary_gabors, special_features, enhanced_g_channel):
    """
    Constructs the 13-dimensional feature vector for every pixel.
    
    Inputs:
    - binary_gabors: Dictionary of 9 binary images from Gabor thresholding.
    - special_features: Dictionary of TH, SC, BPS images.
    - enhanced_g_channel: The CLAHE-enhanced Green channel.
    
    Returns:
    - feature_matrix: (N_pixels, 13) numpy array.
    - image_shape: tuple (height, width) for reconstructing images later.
    """
    print("Constructing 13-dimensional Feature Vector...")
    
    # 1. Collect all feature images in a specific order
    # The paper implies 13 features. We stack them depth-wise.
    feature_list = []
    
    # Add 9 Gabor Binary Images
    # Note: Dictionary order might vary, sorting keys ensures consistency
    for key in sorted(binary_gabors.keys()):
        feature_list.append(binary_gabors[key])
        
    # Add Enhanced G Channel (10th feature) 
    feature_list.append(enhanced_g_channel)
    
    # Add Special Features (TH, SC, BPS) (11th, 12th, 13th)
    feature_list.append(special_features['TH'])
    feature_list.append(special_features['SC'])
    feature_list.append(special_features['BPS'])
    
    # Check consistency
    h, w = enhanced_g_channel.shape
    num_features = len(feature_list) # Should be 13
    
    # Stack and flatten
    # Result shape: (13, H, W) -> Transpose to (H, W, 13) -> Reshape to (N_pixels, 13)
    stack = np.stack(feature_list, axis=-1)
    feature_matrix = stack.reshape(-1, num_features)
    
    # Normalize features to 0-1 range typically helps PCA and Clustering
    # The paper mentions subtracting mean in PCA, but inputs like BPS are 0-255.
    feature_matrix = feature_matrix.astype(np.float32)
    
    return feature_matrix, (h, w)

# --- 2. Principal Component Analysis (PCA) ---
def apply_pca(feature_matrix):
    """
    Applies PCA to decorrelate features.
    
    Paper Analysis of PCA Steps:
    1. "Central point is calculated" (Mean)
    2. "Subtraction of each data from the mean value"
    3. "Decomposition of covariance matrix into eigen-vectors" 
    4. "Vector is multiplied by the original data" 
    5. "None of the feature dimensions are omitted" 
    """
    print("Applying PCA...")
    
    # Initialize PCA keeping all components (n_components=13)
    pca = PCA(n_components=feature_matrix.shape[1])
    
    # Fit and Transform
    # sklearn automatically handles mean subtraction (centering) and eigen decomposition.
    transformed_features = pca.fit_transform(feature_matrix)
    
    return transformed_features

# --- 3. Unsupervised Clustering (Fuzzy C-Means) ---
def fcm_clustering_implementation(data, m=2, max_iter=2000, error=1e-6):
    """
    Manual implementation of Fuzzy C-Means (FCM) to strictly adhere to paper parameters.
    
    Parameters:
    - m = 2 (fuzziness)
    - max_iter = 2000
    - error = 0.000001 (minimum amount of improvement)
    - Clusters = 2 (Vessel vs Non-Vessel)
    """
    print(f"Running FCM Clustering (m={m}, max_iter={max_iter}, eps={error})...")
    
    n_samples, n_features = data.shape
    n_clusters = 2
    
    # Randomly initialize membership matrix U
    # Rows: clusters, Cols: samples
    rng = np.random.default_rng(42)
    U = rng.random((n_clusters, n_samples))
    
    # Normalize U so columns sum to 1
    U = U / np.sum(U, axis=0, keepdims=True)
    
    for iteration in range(max_iter):
        U_prev = U.copy()
        
        # 1. Calculate Cluster Centers (C)
        # C_j = sum(u_ij^m * x_i) / sum(u_ij^m)
        um = U ** m
        centers = (um @ data) / np.sum(um, axis=1, keepdims=True)
        
        # 2. Calculate Distances (Euclidean)
        # distance[j, i] = || x_i - c_j ||
        # Efficient vectorization:
        # We need distance from every point to every center.
        # centers: (2, 13), data: (N, 13)
        # Result dists: (2, N)
        dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2).T
        
        # Avoid division by zero
        dists = np.fmax(dists, 1e-10)
        
        # 3. Update Membership U
        # u_ij = 1 / sum_k ( (d_ij / d_ik) ^ (2/(m-1)) )
        # Power term: 2 / (2 - 1) = 2
        inv_dists = 1.0 / (dists ** 2)
        sum_inv_dists = np.sum(inv_dists, axis=0, keepdims=True)
        U = inv_dists / sum_inv_dists
        
        # 4. Check Convergence
        # "minimum amount of improvement being 0.000001" 
        improvement = np.linalg.norm(U - U_prev)
        if improvement < error:
            print(f"FCM converged at iteration {iteration}")
            break
            
    # Hard Clustering: Assign pixel to cluster with max membership
    labels = np.argmax(U, axis=0)
    
    return labels

def interpret_clusters(labels, shape):
    """
    Identifies which cluster is 'Vessel' and which is 'Non-Vessel'.
    Logic: "The cluster with fewer pixels is the vessel cluster".
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    # Assuming 2 clusters: 0 and 1
    if counts[0] < counts[1]:
        vessel_label = unique[0]
        non_vessel_label = unique[1]
    else:
        vessel_label = unique[1]
        non_vessel_label = unique[0]
        
    # Create Binary Masks
    vessel_mask = (labels == vessel_label).reshape(shape).astype(np.uint8) * 255
    non_vessel_mask = (labels == non_vessel_label).reshape(shape).astype(np.uint8) * 255
    
    return vessel_mask, non_vessel_mask, labels

# --- 4. Supervised Classification & Pipeline Integration ---
def extract_vessels_full_pipeline(rgb_img, processed_channels, binary_gabors, special_features, clf_model=None):
    """
    Combines all steps to produce final vessel extraction.
    
    Args:
        processed_channels: Dict with 'G', 'L', 'Y' (Enhanced).
        binary_gabors: Dict of 9 binary images.
        special_features: Dict of TH, SC, BPS.
        clf_model: A pre-trained sklearn classifier (Decision Tree). 
                   If None, this step is skipped (for demo).
    """
    h, w = processed_channels['G'].shape
    
    # 1. Feature Vector Construction
    X, shape = construct_feature_vector(binary_gabors, special_features, processed_channels['G'])
    
    # 2. PCA
    X_pca = apply_pca(X)
    
    # 3. Clustering (FCM) -> "Thick Vessels"
    labels = fcm_clustering_implementation(X_pca, m=2, max_iter=200, error=1e-6) # reduced max_iter for speed in demo
    vessel_mask_fcm, non_vessel_mask, flat_labels = interpret_clusters(labels, shape)
    
    print("FCM Step Complete. Thick vessels detected.")
    
    # 4. Classification (Decision Tree) -> "Thin Vessels"
    # Applied ONLY to pixels in the non-vessel cluster
    final_vessel_mask = vessel_mask_fcm.copy()
    
    if clf_model is not None:
        print("Running Classification on Non-Vessel Cluster...")
        
        # Indices of pixels that belong to the non-vessel cluster
        non_vessel_indices = np.where(flat_labels == (0 if vessel_mask_fcm.flatten()[0] == 255 else 1))[0]
        
        if len(non_vessel_indices) > 0:
            # Get features for these pixels
            X_subset = X_pca[non_vessel_indices]
            
            # Predict
            # Output is probability map -> threshold -> binary
            # Here we use direct prediction for simplicity, or predict_proba
            preds = clf_model.predict(X_subset)
            
            # Map predictions back to the image
            # If pred is 1 (vessel), set pixel in final mask to 255
            # "Extracted vessel pixels in this step are combined with the vessel cluster" 
            
            # We need to know which class ID in the classifier corresponds to 'vessel'.
            # Usually 1.
            detected_indices = non_vessel_indices[preds == 1]
            
            # Update final mask (Flattened update)
            final_flat = final_vessel_mask.flatten()
            final_flat[detected_indices] = 255
            final_vessel_mask = final_flat.reshape(shape)
            
    else:
        print("No Classifier provided. Skipping supervised step (returning FCM result only).")
        print("To implement fully, train a DecisionTreeClassifier on DRIVE training set pixels.")

    return final_vessel_mask

# ==========================================
# 5. POST-PROCESSING
# ==========================================
def post_processing(vessel_mask, fov_mask):
    """
    Removes artifacts at the border of the FOV.
    Method: Erode FOV mask and apply bitwise AND to remove false positives at the edge.
    Corresponds to paper Section 3.4.
    """
    # "radius of the original FOV mask is reduced by some pixels"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded_fov = cv2.erode(fov_mask, kernel, iterations=1)
    
    # "The new mask is applied on the output image"
    clean_mask = cv2.bitwise_and(vessel_mask, vessel_mask, mask=eroded_fov)
    return clean_mask

# ==========================================
# 6. TRAINING UTILITY (Updated for 21-40 Naming)
# ==========================================
def load_image_safe(path, grayscale=False):
    """
    Robust image loader that handles .gif files (common in DRIVE dataset)
    which cv2.imread often fails to load.
    """
    # Try OpenCV first
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        
    # If OpenCV failed (likely due to .gif extension), use PIL
    if img is None:
        try:
            pil_img = Image.open(path)
            if grayscale:
                img = np.array(pil_img.convert('L'))
            else:
                # Convert RGB (PIL) to BGR (OpenCV standard)
                rgb = np.array(pil_img.convert('RGB'))
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Warning: Failed to load {path}. Error: {e}")
            return None
    return img

def calculate_metrics(prediction, ground_truth, fov_mask=None):
    """
    Calculates metrics according to Section 4.2 of the paper.
    
    Paper Specifications:
    - "evaluation metrics are calculated only for pixels inside the FOV" 
    - Acc: Eq. 2 [cite: 419]
    - Sen: Eq. 3 [cite: 432]
    - Spe: Eq. 4 [cite: 438]
    - PPV: Eq. 5 [cite: 442]
    """
    # flatten arrays
    p = prediction.flatten()
    t = ground_truth.flatten()
    
    # Apply FOV masking if provided 
    if fov_mask is not None:
        m = fov_mask.flatten()
        # Keep only pixels inside the FOV (where mask is non-zero)
        p = p[m > 0]
        t = t[m > 0]

    # Binarize (ensure 0 or 1)
    p = (p > 127).astype(int)
    t = (t > 127).astype(int)
    
    # Confusion Matrix
    # labels=[0, 1] ensures we get a 2x2 matrix even if one class is missing
    tn, fp, fn, tp = confusion_matrix(t, p, labels=[0, 1]).ravel()
    
    # Avoid division by zero with epsilon
    eps = 1e-10
    
    # Calculate Metrics based on Paper Eqs.
    sen = tp / (tp + fn + eps)             # Sensitivity [cite: 432]
    spe = tn / (tn + fp + eps)             # Specificity [cite: 438]
    acc = (tp + tn) / (tp + tn + fp + fn + eps) # Accuracy [cite: 419]
    ppv = tp / (tp + fp + eps)             # Positive Predictive Value [cite: 442]
    
    return sen, spe, acc, ppv


def train_classifier_module(train_img_dir, train_mask_dir, train_manual_dir):
    """
    Trains the Decision Tree classifier using the Training Set (Images 21-40).
    """
    print("\n--- Starting Training Phase ---")
    X_train = []
    y_train = []
    
    # Load and Sort files to ensure alignment (21 aligns with 21, etc.)
    # Note: We rely on the fact that files start with "21_...", "22_..." etc.
    img_paths = sorted(glob.glob(os.path.join(train_img_dir, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(train_mask_dir, "*.gif")))
    manual_paths = sorted(glob.glob(os.path.join(train_manual_dir, "*.gif")))
    
    print(f"Found {len(img_paths)} training images.")
    
    # Safety Check
    if len(img_paths) == 0:
        print("Error: No training images found in", train_img_dir)
        return None
        
    # Iterate through training images
    # Using zip ensures we process the matched triplet (Image, Mask, Manual)
    for i, (img_p, mask_p, manual_p) in enumerate(zip(img_paths, mask_paths, manual_paths)):
        fname = os.path.basename(img_p)
        print(f"Training on Image {i+1}: {fname}")
        
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, 0)
        gt = cv2.imread(manual_p, 0)
        
        if img is None or mask is None or gt is None:
            print(f"Skipping {fname} due to load error.")
            continue
        
        # 1. Preprocessing
        rgb_c, _, enh_channels = preprocess_pipeline(img_p, mask_p)
        
        # Get BBox for cropping Ground Truth (GT)
        bbox = get_fov_bbox(mask)
        gt_crop = crop_to_fov(gt, bbox)
        
        # 2. Extract Features
        gabor_feats = apply_multiscale_gabor(enh_channels)
        bin_gabor = automatic_thresholding(gabor_feats)
        special = extract_special_features(enh_channels['G'])
        
        # 3. Construct Vector
        features, _ = construct_feature_vector(bin_gabor, special, enh_channels['G'])
        
        # 4. Apply PCA
        pca = PCA(n_components=13)
        feat_pca = pca.fit_transform(features)
        
        # 5. Prepare Labels (0 or 1)
        labels = (gt_crop.flatten() > 127).astype(int)
        
        # 6. Subsample (10k pixels per image)
        num_samples = 10000
        if len(labels) > num_samples:
            idx = np.random.choice(len(labels), num_samples, replace=False)
            X_train.append(feat_pca[idx])
            y_train.append(labels[idx])
        else:
            X_train.append(feat_pca)
            y_train.append(labels)
            
    if not X_train:
        print("Error: Training data empty.")
        return None
        
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    
    print(f"Training Decision Tree on {len(y_train)} pixels...")
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    print("Training Complete.")
    return clf

# ... [Keep all your existing helper functions: get_fov_bbox, preprocess_pipeline, etc.] ...
# ... [Keep feature extraction functions: apply_multiscale_gabor, extract_special_features, etc.] ...
# ... [Keep pipeline functions: extract_vessels_full_pipeline, post_processing, etc.] ...

# ==========================================
# 7. VISUALIZATION (Updated to Save to Folder)
# ==========================================
def plot_results(rgb_crop, orig_ch, enh_ch, gab_f, bin_g, spec_f, fcm_res, cls_res, final_res, img_name, save_dir):
    """
    Generates Figures 3-12 and saves them to the specified directory.
    """
    print(f"Saving figures for {img_name} to {save_dir}...")

    # --- Fig 3: Selected three channels ---
    # Caption: "(a) G channel, (b) Y channel, and (c) L channel"
    fig3, ax3 = plt.subplots(1, 3, figsize=(12, 4))
    fig3.suptitle(f"Fig 3: Selected Channels - {img_name}")
    ax3[0].imshow(orig_ch['G'], 'gray'); ax3[0].set_title("(a) G Channel"); ax3[0].axis('off')
    ax3[1].imshow(orig_ch['Y'], 'gray'); ax3[1].set_title("(b) Y Channel"); ax3[1].axis('off')
    ax3[2].imshow(orig_ch['L'], 'gray'); ax3[2].set_title("(c) L Channel"); ax3[2].axis('off')
    fig3.savefig(os.path.join(save_dir, f"Fig3_Selected_Channels_{img_name}.png"))
    plt.close(fig3)

    # --- Fig 4: Histogram Equalization Output ---
    # Caption: "Outputs of the histogram equalization operation..."
    fig4, ax4 = plt.subplots(1, 3, figsize=(12, 4))
    fig4.suptitle(f"Fig 4: CLAHE Outputs - {img_name}")
    ax4[0].imshow(enh_ch['G'], 'gray'); ax4[0].set_title("(a) Enhanced G"); ax4[0].axis('off')
    ax4[1].imshow(enh_ch['Y'], 'gray'); ax4[1].set_title("(b) Enhanced Y"); ax4[1].axis('off')
    ax4[2].imshow(enh_ch['L'], 'gray'); ax4[2].set_title("(c) Enhanced L"); ax4[2].axis('off')
    fig4.savefig(os.path.join(save_dir, f"Fig4_CLAHE_Outputs_{img_name}.png"))
    plt.close(fig4)

    # --- Fig 5: Gabor Filter Outputs ---
    # Caption: "Outputs of the Gabor filters on (first row) G, (second row) Y, and (third row) L channels"
    fig5, ax5 = plt.subplots(3, 3, figsize=(12, 10))
    fig5.suptitle(f"Fig 5: Gabor Outputs - {img_name}")
    
    # Row 1: G Channel
    ax5[0, 0].imshow(gab_f['G_9'], 'gray');  ax5[0, 0].set_title("G (Wave 9)");  ax5[0, 0].axis('off')
    ax5[0, 1].imshow(gab_f['G_10'], 'gray'); ax5[0, 1].set_title("G (Wave 10)"); ax5[0, 1].axis('off')
    ax5[0, 2].imshow(gab_f['G_11'], 'gray'); ax5[0, 2].set_title("G (Wave 11)"); ax5[0, 2].axis('off')

    # Row 2: Y Channel
    ax5[1, 0].imshow(gab_f['Y_9'], 'gray');  ax5[1, 0].set_title("Y (Wave 9)");  ax5[1, 0].axis('off')
    ax5[1, 1].imshow(gab_f['Y_10'], 'gray'); ax5[1, 1].set_title("Y (Wave 10)"); ax5[1, 1].axis('off')
    ax5[1, 2].imshow(gab_f['Y_11'], 'gray'); ax5[1, 2].set_title("Y (Wave 11)"); ax5[1, 2].axis('off')

    # Row 3: L Channel
    ax5[2, 0].imshow(gab_f['L_9'], 'gray');  ax5[2, 0].set_title("L (Wave 9)");  ax5[2, 0].axis('off')
    ax5[2, 1].imshow(gab_f['L_10'], 'gray'); ax5[2, 1].set_title("L (Wave 10)"); ax5[2, 1].axis('off')
    ax5[2, 2].imshow(gab_f['L_11'], 'gray'); ax5[2, 2].set_title("L (Wave 11)"); ax5[2, 2].axis('off')

    plt.tight_layout()
    fig5.savefig(os.path.join(save_dir, f"Fig5_Gabor_Outputs_{img_name}.png"))
    plt.close(fig5)

    # --- Fig 6: Automatic Thresholding Results ---
    # Caption: "Results of the automatic thresholding..."
    fig6, ax6 = plt.subplots(3, 3, figsize=(12, 10))
    fig6.suptitle(f"Fig 6: Automatic Thresholding - {img_name}")

    # Row 1: G Channel
    ax6[0, 0].imshow(bin_g['G_9'], 'gray');  ax6[0, 0].set_title("Binary G_9");  ax6[0, 0].axis('off')
    ax6[0, 1].imshow(bin_g['G_10'], 'gray'); ax6[0, 1].set_title("Binary G_10"); ax6[0, 1].axis('off')
    ax6[0, 2].imshow(bin_g['G_11'], 'gray'); ax6[0, 2].set_title("Binary G_11"); ax6[0, 2].axis('off')

    # Row 2: Y Channel
    ax6[1, 0].imshow(bin_g['Y_9'], 'gray');  ax6[1, 0].set_title("Binary Y_9");  ax6[1, 0].axis('off')
    ax6[1, 1].imshow(bin_g['Y_10'], 'gray'); ax6[1, 1].set_title("Binary Y_10"); ax6[1, 1].axis('off')
    ax6[1, 2].imshow(bin_g['Y_11'], 'gray'); ax6[1, 2].set_title("Binary Y_11"); ax6[1, 2].axis('off')

    # Row 3: L Channel
    ax6[2, 0].imshow(bin_g['L_9'], 'gray');  ax6[2, 0].set_title("Binary L_9");  ax6[2, 0].axis('off')
    ax6[2, 1].imshow(bin_g['L_10'], 'gray'); ax6[2, 1].set_title("Binary L_10"); ax6[2, 1].axis('off')
    ax6[2, 2].imshow(bin_g['L_11'], 'gray'); ax6[2, 2].set_title("Binary L_11"); ax6[2, 2].axis('off')

    plt.tight_layout()
    fig6.savefig(os.path.join(save_dir, f"Fig6_Auto_Thresholding_{img_name}.png"))
    plt.close(fig6)

    # --- Fig 7: Top Hat (TH) ---
    # Caption: "(a) Input image, (b) extracted TH feature"
    fig7, ax7 = plt.subplots(1, 2, figsize=(8, 4))
    fig7.suptitle(f"Fig 7: Top Hat (TH) - {img_name}")
    ax7[0].imshow(rgb_crop); ax7[0].set_title("(a) Input"); ax7[0].axis('off')
    ax7[1].imshow(spec_f['TH'], 'gray'); ax7[1].set_title("(b) TH Feature"); ax7[1].axis('off')
    fig7.savefig(os.path.join(save_dir, f"Fig7_TopHat_{img_name}.png"))
    plt.close(fig7)

    # --- Fig 8: Shade Correction (SC) ---
    # Caption: "(a) Input image, (b) extracted SC feature"
    fig8, ax8 = plt.subplots(1, 2, figsize=(8, 4))
    fig8.suptitle(f"Fig 8: Shade Correction (SC) - {img_name}")
    ax8[0].imshow(rgb_crop); ax8[0].set_title("(a) Input"); ax8[0].axis('off')
    ax8[1].imshow(spec_f['SC'], 'gray'); ax8[1].set_title("(b) SC Feature"); ax8[1].axis('off')
    fig8.savefig(os.path.join(save_dir, f"Fig8_Shade_Correction_{img_name}.png"))
    plt.close(fig8)
    
# --- Fig 9: Bit Plane Slicing (BPS) ---
    # Caption: "(a) Input, (b) 1st... (i) 8th, (j) Sum" 
    # We need 10 subplots. Let's do 2 rows of 5 columns.
    fig9, ax9 = plt.subplots(2, 5, figsize=(15, 6))
    fig9.suptitle(f"Fig 9: BPS Feature (Bit Planes 1-8 and Sum) - {img_name}")
    
    # Flatten axes for easy iteration (0 to 9)
    axes = ax9.flatten()
    
    # (a) Input Image
    axes[0].imshow(rgb_crop)
    axes[0].set_title("(a) Input")
    
    # (b)-(i) Bit Planes 1 through 8
    planes = spec_f['BPS_planes'] # List of 8 images
    for i in range(8):
        # Place plane i in subplot index i+1
        axes[i+1].imshow(planes[i], 'gray')
        # char(98) is 'b', char(99) is 'c', etc.
        axes[i+1].set_title(f"({chr(98+i)}) {i+1}th Plane")
        
    # (j) Sum of Last Two
    axes[9].imshow(spec_f['BPS'], 'gray')
    axes[9].set_title("(j) Sum (7th+8th)")
    
    # Turn off axis labels for all subplots
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    fig9.savefig(os.path.join(save_dir, f"Fig9_BPS_{img_name}.png"))
    plt.close(fig9)

    # --- Fig 10: Unsupervised Clustering (FCM) ---
    # Caption: "(a) Input image, (b) initial extraction... by clustering"
    fig10, ax10 = plt.subplots(1, 2, figsize=(8, 4))
    fig10.suptitle(f"Fig 10: Unsupervised Clustering (FCM) - {img_name}")
    ax10[0].imshow(rgb_crop); ax10[0].set_title("(a) Input"); ax10[0].axis('off')
    ax10[1].imshow(fcm_res, 'gray'); ax10[1].set_title("(b) FCM Output"); ax10[1].axis('off')
    fig10.savefig(os.path.join(save_dir, f"Fig10_FCM_Clustering_{img_name}.png"))
    plt.close(fig10)

    # --- Fig 11: Classification Step ---
    # Caption: "(a) Input... (b) clustering... (c) all extracted vessels..."
    fig11, ax11 = plt.subplots(1, 3, figsize=(12, 4))
    fig11.suptitle(f"Fig 11: Classification Step - {img_name}")
    ax11[0].imshow(rgb_crop); ax11[0].set_title("(a) Input"); ax11[0].axis('off')
    ax11[1].imshow(fcm_res, 'gray'); ax11[1].set_title("(b) FCM Output"); ax11[1].axis('off')
    ax11[2].imshow(cls_res, 'gray'); ax11[2].set_title("(c) FCM + Classification"); ax11[2].axis('off')
    fig11.savefig(os.path.join(save_dir, f"Fig11_Classification_{img_name}.png"))
    plt.close(fig11)

    # --- Fig 12: Final Result (Post-Processing) ---
    # Caption: "(a) Input... (b) vessel extraction... (c) final extracted vessels..."
    fig12, ax12 = plt.subplots(1, 3, figsize=(12, 4))
    fig12.suptitle(f"Fig 12: Final Result - {img_name}")
    ax12[0].imshow(rgb_crop); ax12[0].set_title("(a) Input"); ax12[0].axis('off')
    ax12[1].imshow(cls_res, 'gray'); ax12[1].set_title("(b) Vessel Extraction"); ax12[1].axis('off')
    ax12[2].imshow(final_res, 'gray'); ax12[2].set_title("(c) Final Result"); ax12[2].axis('off')
    fig12.savefig(os.path.join(save_dir, f"Fig12_Final_Result_{img_name}.png"))
    plt.close(fig12)

# ==========================================
# 8. MAIN EXECUTION (Updated Paths & Folder Creation)
# ==========================================

def main():
    # --- Configuration ---
    base_dir = "DRIVE"
    train_dir = os.path.join(base_dir, "training")
    test_dir = os.path.join(base_dir, "test")
    
    # NEW: Create a folder to save the plotted figures
    save_dir = os.path.join(base_dir, "Paper_Figures_Output")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Info] Figures will be saved to: {save_dir}")

    # 1. Train the Classifier
    # Explicitly pointing to subfolders
    clf_model = train_classifier_module(
        os.path.join(train_dir, "images"),
        os.path.join(train_dir, "mask"),
        os.path.join(train_dir, "1st_manual")
    )
    
    if clf_model is None:
        print("Classifier training failed. Exiting.")
        return

    # 2. Process Test Images (01-20)
    print("\n--- Starting Testing Phase ---")
    test_imgs = sorted(glob.glob(os.path.join(test_dir, "images", "*.tif")))
    test_masks = sorted(glob.glob(os.path.join(test_dir, "mask", "*.gif")))
    test_gts = sorted(glob.glob(os.path.join(test_dir, "1st_manual", "*.gif")))
    
    if len(test_imgs) == 0:
        print("Error: No test images found in", os.path.join(test_dir, "images"))
        return

    metrics_list = []
    
    # Zip ensuring we iterate matched files
    for i, (img_path, mask_path) in enumerate(zip(test_imgs, test_masks)):
        img_name = os.path.basename(img_path)
        print(f"Processing Test Image {i+1}: {img_name}")
        
        # A. Preprocessing Pipeline
        rgb_c, orig_ch, enh_ch = preprocess_pipeline(img_path, mask_path)
        
        # B. Feature Extraction
        gabor_feats = apply_multiscale_gabor(enh_ch)
        bin_gabor = automatic_thresholding(gabor_feats)
        special = extract_special_features(enh_ch['G'])
        
        # C. Full Extraction
        # C1: Vector Construction
        features, shape = construct_feature_vector(bin_gabor, special, enh_ch['G'])
        X_pca = apply_pca(features)
        
        # C2: Clustering (FCM)
        labels = fcm_clustering_implementation(X_pca, m=2, max_iter=100, error=1e-5)
        fcm_mask, non_vessel_mask, flat_labels = interpret_clusters(labels, shape)
        
        # C3: Classification
        cls_mask = fcm_mask.copy()
        if clf_model:
            # Simple heuristic for classifying non-vessel pixels
            # We identify pixels in the "background" cluster (value 0)
            non_vessel_idx = np.where(fcm_mask.flatten() == 0)[0]

            if len(non_vessel_idx) > 0:
                X_subset = X_pca[non_vessel_idx]
                preds = clf_model.predict(X_subset)
                
                # Detected vessels (1)
                detected_idx = non_vessel_idx[preds == 1]
                
                # Update
                flat_mask = cls_mask.flatten()
                flat_mask[detected_idx] = 255
                cls_mask = flat_mask.reshape(shape)

        # D. Post-Processing
        mask_orig = cv2.imread(mask_path, 0)
        bbox = get_fov_bbox(mask_orig)
        mask_crop = crop_to_fov(mask_orig, bbox)
        final_mask = post_processing(cls_mask, mask_crop)
        
        # E. Visualization and Saving (Modified: Now runs for ALL images)
        # ------------------------------------------------------------------
        # REMOVED: if i == 0:
        # ------------------------------------------------------------------
        plot_results(rgb_c, orig_ch, enh_ch, gabor_feats, bin_gabor, special, 
                     fcm_mask, cls_mask, final_mask, img_name, save_dir)
            
        # F. Evaluation (Table 1)
        if i < len(test_gts):
            # Use load_image_safe for GT (handles .gif)
            gt = load_image_safe(test_gts[i], grayscale=True)
            
            if gt is not None:
                gt_crop = crop_to_fov(gt, bbox)
                
                # ERROR FIX: Unpack 4 values instead of 3
                # Pass mask_crop as fov_mask for correct calculation inside FOV
                sen, spe, acc, ppv = calculate_metrics(final_mask, gt_crop, fov_mask=mask_crop)
                
                # Append all 4 metrics
                metrics_list.append([img_name, sen, spe, acc, ppv])
            else:
                print(f"  Warning: Could not load GT {test_gts[i]}")

    # 3. Print Table 1 (Updated to include PPV)
    print("\n" + "="*70)
    print("Table 1: Performance Measures (Calculated on Test Set)")
    print("="*70)
    # Added PPV column to header
    print(f"{'Image':<20} | {'Sen':<10} | {'Spe':<10} | {'Acc':<10} | {'PPV':<10}")
    print("-" * 70)
    
    if not metrics_list:
        print("No metrics. Check '1st_manual' folder in 'test' directory.")
    else:
        avg_sen, avg_spe, avg_acc, avg_ppv = 0, 0, 0, 0
        for m in metrics_list:
            # Updated row printing for 4 metrics
            print(f"{m[0]:<20} | {m[1]:.4f}     | {m[2]:.4f}     | {m[3]:.4f}     | {m[4]:.4f}")
            avg_sen += m[1]
            avg_spe += m[2]
            avg_acc += m[3]
            avg_ppv += m[4]
            
        n = len(metrics_list)
        print("-" * 70)
        # Updated average printing
        print(f"{'AVERAGE':<20} | {avg_sen/n:.4f}     | {avg_spe/n:.4f}     | {avg_acc/n:.4f}     | {avg_ppv/n:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()