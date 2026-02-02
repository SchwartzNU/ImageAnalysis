import os
import numpy as np
import pandas as pd
import dearpygui.dearpygui as dpg
import nd2
import cv2
from skimage import exposure, measure
from scipy.ndimage import center_of_mass
from scipy.stats import skew
from scipy.signal import find_peaks
from skimage.measure import label, regionprops
from cellpose import models, denoise
import GUI_helpers

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

trace_data_df = None

def assign_colors_to_masks(mask_array):
    """
    Assign colors to masks using graph coloring to maximize visual differences between adjacent cells.
    Returns: dict mapping mask_id -> (r, g, b) normalized to 0-1 range
    """
    from collections import defaultdict
    
    unique_masks = np.unique(mask_array)
    unique_masks = unique_masks[unique_masks > 0]  # Exclude background
    
    if len(unique_masks) == 0:
        return {}
    
    # Define a palette of highly distinct colors using proper HSV space
    palette = []
    # OpenCV HSV: Hue 0-180, Saturation 0-255, Value 0-255
    # Generate many distinct colors with different hues, saturations, and values
    hues = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]  # 12 hues in 0-180 range (multiply by 180/360)
    # Define a palette sized to the number of masks (avoid running out of distinct hues)
    palette = []
    # OpenCV HSV: Hue 0-180, Saturation 0-255, Value 0-255
    palette_size = max(72, len(unique_masks))
    # Evenly spaced hues across 0-179
    hues = np.linspace(0, 179, palette_size, endpoint=False).astype(int)
    sat = 255
    val = 255
    for hue in hues:
        hsv_color = np.uint8([[[int(hue), sat, val]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
        r = float(rgb_color[0, 0, 0]) / 255.0
        g = float(rgb_color[0, 0, 1]) / 255.0
        b = float(rgb_color[0, 0, 2]) / 255.0
        palette.append((r, g, b))
        print(f"[DEBUG] Palette color range check:")
        rs = [c[0] for c in palette]
        gs = [c[1] for c in palette]
        bs = [c[2] for c in palette]
        print(f"[DEBUG]   R range: {min(rs):.2f} - {max(rs):.2f}")
        print(f"[DEBUG]   G range: {min(gs):.2f} - {max(gs):.2f}")
        print(f"[DEBUG]   B range: {min(bs):.2f} - {max(bs):.2f}")
    
    # Build adjacency graph - masks are adjacent if they touch
    adjacencies = defaultdict(set)
    for mask_id in unique_masks:
        mask = (mask_array == mask_id).astype(np.uint8)
        # Dilate slightly to find neighbors
        dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        # Find which masks overlap with dilated region
        for other_id in unique_masks:
            if other_id != mask_id:
                other_mask = (mask_array == other_id).astype(np.uint8)
                if np.any(dilated & other_mask):  # Both are uint8, bitwise AND works
                    adjacencies[mask_id].add(other_id)
    
    # Improved greedy assignment: prefer palette colors that maximize color-distance
    # to already-assigned neighboring colors so we use many distinct colors while
    # still avoiding exact adjacency matches.
    color_assignment = {}
    sorted_masks = sorted(unique_masks, key=lambda x: -len(adjacencies[x]))

    def color_distance(c1, c2):
        return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2

    for mask_id in sorted_masks:
        # Colors already used by adjacent masks
        adjacent_colors = [color_assignment[adj] for adj in adjacencies[mask_id] if adj in color_assignment]

        best_color = None
        best_score = -1.0

        # Evaluate each candidate in palette and pick the one maximizing the
        # minimal distance to adjacent colors (so it's as different as possible)
        for color in palette:
            if color in adjacent_colors:
                continue
            if not adjacent_colors:
                # No assigned neighbors yet -> prefer colors that are not yet used globally
                # Score by inverse of how many times color already used
                used_count = sum(1 for v in color_assignment.values() if v == color)
                score = 1.0 / (1 + used_count)
            else:
                # Score is the minimal squared distance to neighbors
                dists = [color_distance(color, ac) for ac in adjacent_colors]
                score = min(dists) if dists else 0.0

            if score > best_score:
                best_score = score
                best_color = color

        if best_color is None:
            # Fallback: pick a palette color with minimal global usage
            counts = {c: sum(1 for v in color_assignment.values() if v == c) for c in palette}
            best_color = min(counts.keys(), key=lambda c: counts[c])

        color_assignment[mask_id] = best_color

    # Debug: Check what colors got assigned
    if len(color_assignment) > 0:
        assigned_colors = set(color_assignment.values())
        print(f"[DEBUG] Color assignment: {len(color_assignment)} masks, {len(assigned_colors)} unique colors")
        if len(assigned_colors) > 1:
            colors_list = list(assigned_colors)[:5]
            print(f"[DEBUG] Sample assigned colors: {colors_list}")

    return color_assignment

def save_segmentation_visualization(mask_array, color_assignment, output_path):
    """
    Save segmentation visualization with labels as a multi-page TIFF stack.
    """
    seg_viz = create_labeled_segmentation_image(mask_array, color_assignment)
    # Convert from RGBA float32 to RGB uint8 for saving
    rgb_uint8 = (seg_viz[..., :3] * 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_uint8)
    print(f"Saved segmentation visualization to: {output_path}")

def create_filtered_segmentation_visualization(mask_array, color_assignment, filtered_idxs, output_path=None):
    """
    Create segmentation visualization where removed cells are transparent/hidden.
    filtered_idxs: list of mask IDs that passed filtering (1-indexed in mask_array)
    output_path: optional path to save PNG. If None, only returns RGBA array.
    Returns: RGBA image as numpy array
    """
    h, w = mask_array.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    
    unique_masks = np.unique(mask_array)
    unique_masks = unique_masks[unique_masks > 0]
    
    # Draw only filtered masks (removed cells stay transparent)
    for mask_id in unique_masks:
        if (mask_id - 1) not in filtered_idxs:
            continue  # Skip removed cells
            
        mask = (mask_array == mask_id)
        if mask_id in color_assignment:
            r, g, b = color_assignment[mask_id]
            rgba[mask, 0] = r
            rgba[mask, 1] = g
            rgba[mask, 2] = b
            rgba[mask, 3] = 1.0  # Fully opaque
    
    # Add labels for filtered cells only
    for mask_id in unique_masks:
        if (mask_id - 1) not in filtered_idxs:
            continue  # Skip removed cells
            
        mask = (mask_array == mask_id).astype(np.uint8)
        
        # Find centroid for label
        props = cv2.moments(mask)
        if props['m00'] != 0:
            cx = int(props['m10'] / props['m00'])
            cy = int(props['m01'] / props['m00'])
            
            # Get background color
            if mask_id in color_assignment:
                r, g, b = color_assignment[mask_id]
            else:
                r, g, b = 0.5, 0.5, 0.5
            
            # Determine text color based on background brightness
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = (1.0, 1.0, 1.0) if brightness < 0.5 else (0.0, 0.0, 0.0)
            text_color_bgr = (int(text_color[2] * 255), int(text_color[1] * 255), int(text_color[0] * 255))
            
            # Create label text
            label_text = str(int(mask_id))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size
            text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_w, text_h = text_size
            
            # Draw text
            x = max(0, min(cx - text_w // 2, w - text_w))
            y = max(text_h, min(cy + text_h // 2, h))
            
            text_img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(text_img, label_text, (x, y), font, font_scale, text_color_bgr, thickness)
            
            # Blend text into RGBA
            text_mask = np.any(text_img != 0, axis=2)
            rgba[text_mask, 0] = text_color[0]
            rgba[text_mask, 1] = text_color[1]
            rgba[text_mask, 2] = text_color[2]
            rgba[text_mask, 3] = 1.0
    
    # Save as PNG if output_path provided
    if output_path:
        rgb_uint8 = (rgba[..., :3] * 255).astype(np.uint8)
        bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr_uint8)
        print(f"Saved filtered segmentation visualization to: {output_path}")
    
    return rgba

def create_labeled_segmentation_image(mask_array, color_assignment, filtered_idxs=None, label_min_area=5, force_labels=False, show_removed=False):
    """
    Create a segmentation visualization with colored, filled masks and numeric labels.
    Text color is chosen (black or white) based on background brightness.
    Returns: RGBA image as numpy array
    """
    h, w = mask_array.shape
    print(f"[DEBUG] create_labeled_segmentation_image: {len(color_assignment)} colors, mask shape {h}x{w}")
    # Print sample colors
    if color_assignment:
        sample_ids = list(color_assignment.keys())[:3]
        print(f"[DEBUG] Sample color assignments: {[(mid, color_assignment[mid]) for mid in sample_ids]}")
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    
    unique_masks = np.unique(mask_array)
    unique_masks = unique_masks[unique_masks > 0]

        # Draw filled colored masks. If filtered_idxs is provided and show_removed is True, render removed masks in semi-transparent gray.
    for mask_id in unique_masks:
        mask = (mask_array == mask_id)
        kept = True
        if filtered_idxs is not None:
            # filtered_idxs stores zero-based indices of masks that were kept
            kept = ((mask_id - 1) in filtered_idxs)

        if kept:
            if mask_id in color_assignment:
                r, g, b = color_assignment[mask_id]
            else:
                r, g, b = 0.5, 0.5, 0.5
            rgba[mask, 0] = r
            rgba[mask, 1] = g
            rgba[mask, 2] = b
            rgba[mask, 3] = 1.0  # Fully opaque
        else:
            if show_removed:
                # draw removed as semi-transparent gray
                rgba[mask, 0] = 0.25
                rgba[mask, 1] = 0.25
                rgba[mask, 2] = 0.25
                rgba[mask, 3] = 0.35
    
    # Add labels with dynamic text color. Use regionprops centroid as a robust
    # fallback and skip extremely small regions unless force_labels is True.
    from skimage.measure import regionprops
    for mask_id in unique_masks:
        mask_bool = (mask_array == mask_id)
        mask = mask_bool.astype(np.uint8)

        # Use regionprops for robust measurements
        props = regionprops(mask)
        if not props:
            continue
        prop = props[0]
        area = prop.area
        # Skip labeling very small regions unless forced
        if (not force_labels) and area < label_min_area:
            continue

        # Centroid (row, col)
        cyf, cxf = prop.centroid
        cx = int(round(cxf))
        cy = int(round(cyf))

        # Get background color for this mask
        if mask_id in color_assignment:
            r, g, b = color_assignment[mask_id]
        else:
            r, g, b = 0.5, 0.5, 0.5

        # Determine text color based on background brightness
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (1.0, 1.0, 1.0) if brightness < 0.5 else (0.0, 0.0, 0.0)
        text_color_bgr = (int(text_color[2] * 255), int(text_color[1] * 255), int(text_color[0] * 255))

        # Create label text
        label_text = str(int(mask_id))
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Scale font with area so larger masks get larger text
        font_scale = 0.5 if area < 200 else 0.8
        thickness = 2 if area >= 200 else 1

        # Determine bounding box for this region and adapt font to fit within it
        minr, minc, maxr, maxc = prop.bbox
        bbox_w = max(1, maxc - minc)
        bbox_h = max(1, maxr - minr)

        # Start with base font scale and reduce until it fits within the bbox width
        text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_w, text_h = text_size
        max_text_w = max(1, bbox_w - 4)
        while text_w > max_text_w and font_scale > 0.2:
            font_scale -= 0.1
            text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_w, text_h = text_size

        # Place text centered in the region's bbox and clamp to image bounds
        cx_clamped = int(min(max(cx, minc + text_w // 2), maxc - text_w // 2))
        cy_clamped = int(min(max(cy, minr + text_h // 2), maxr - text_h // 2))
        x = max(0, min(cx_clamped - text_w // 2, w - text_w))
        y = max(text_h, min(cy_clamped + text_h // 2, h))

        # Create a temporary image for text rendering
        text_img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(text_img, label_text, (x, y), font, font_scale, text_color_bgr, thickness)

        # Blend text into RGBA
        text_mask = np.any(text_img != 0, axis=2)
        rgba[text_mask, 0] = text_color[0]
        rgba[text_mask, 1] = text_color[1]
        rgba[text_mask, 2] = text_color[2]
        rgba[text_mask, 3] = 1.0
    
    return rgba

def auto_brightness_contrast(image):
    normalized_image = image.astype(np.float32) / 255.0
    equalized_image = exposure.equalize_adapthist(normalized_image)
    equalized_image = (equalized_image * 255).astype(np.uint8)
    return equalized_image

def to_8bit(stack):
    stack = stack.astype(np.float32)
    stack -= np.min(stack)
    if np.max(stack) != 0:
        stack /= np.max(stack)
    return (255 * stack).astype(np.uint8)

def extract_masks(total_masks, points, reset_mask_ids=True):
    mod_masks = total_masks.copy()

    if isinstance(points, int):
        points = np.array([points])
    else:
        points = np.array(points)

    points = points + 1 

    mod_masks[~np.isin(mod_masks, points)] = 0

    if reset_mask_ids:
        unique_values = np.unique(mod_masks)
        unique_values.sort()
        for new_id, old_id in enumerate(unique_values):
            mod_masks[mod_masks == old_id] = new_id

    return mod_masks

def get_mask_diameter(mask):
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] == 0:
        return 15
    d = np.max(np.ptp(coords, axis=0))
    return max(5, d)

def square_mask(mask, perc_increase: int = 40):
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)

    # If no regions found, fall back to bounding box of nonzero pixels or centered square
    if not regions:
        coords = np.column_stack(np.where(mask > 0))
        h, w = mask.shape
        if coords.shape[0] == 0:
            side = max(5, min(h, w) // 2)
            cy, cx = h // 2, w // 2
            half = side // 2
            new_mask = np.zeros_like(mask, dtype=np.uint8)
            r0, r1 = max(0, cy - half), min(h, cy + half)
            c0, c1 = max(0, cx - half), min(w, cx + half)
            new_mask[r0:r1, c0:c1] = 1
            return new_mask
        else:
            minr, minc = coords.min(axis=0)
            maxr, maxc = coords.max(axis=0)
            new_mask = np.zeros_like(mask, dtype=np.uint8)
            new_mask[minr:maxr + 1, minc:maxc + 1] = 1
            return new_mask

    largest_region = max(regions, key=lambda r: r.area)
    min_row, min_col, max_row, max_col = largest_region.bbox
    centroid = largest_region.centroid

    height = max_row - min_row
    width = max_col - min_col
    diameter = max(height, width)
    new_diameter = int(diameter * (1 + perc_increase / 100.0))
    if new_diameter <= 0:
        new_diameter = max(height, width, 5)
    half_side = new_diameter // 2

    cy = int(round(centroid[0]))
    cx = int(round(centroid[1]))
    top = max(cy - half_side, 0)
    left = max(cx - half_side, 0)
    bottom = min(cy + half_side, mask.shape[0])
    right = min(cx + half_side, mask.shape[1])

    new_mask = np.zeros_like(mask, dtype=np.uint8)
    if bottom > top and right > left:
        new_mask[top:bottom, left:right] = 1
    return new_mask

def get_sq_stacks(image, single_mask):
    sq_maski = square_mask(single_mask)

    props = regionprops(sq_maski.astype(int))
    if not props:
        min_row, min_col, max_row, max_col = 0, 0, sq_maski.shape[0], sq_maski.shape[1]
    else:
        min_row, min_col, max_row, max_col = props[0].bbox

    # Clip to image bounds
    y_max, x_max = image.shape[2], image.shape[3]
    min_row = max(0, min_row)
    min_col = max(0, min_col)
    max_row = min(y_max, max_row)
    max_col = min(x_max, max_col)

    sq_DAPI_stack = image[:, 0, min_row:max_row, min_col:max_col]
    sq_eGFP_stack = image[:, 1, min_row:max_row, min_col:max_col]
    sq_WGA_stack = image[:, 2, min_row:max_row, min_col:max_col]
    sq_GLUT1_stack = image[:, 3, min_row:max_row, min_col:max_col]

    sq_stacks = np.stack((sq_DAPI_stack, sq_eGFP_stack, sq_WGA_stack, sq_GLUT1_stack))

    return sq_stacks

def nucleus_com(single_channel, mask):
    masked_channel = single_channel * mask

    z_prof = np.sum(masked_channel, axis=(1, 2))
    z_max_idx = np.argmax(z_prof)

    com = center_of_mass(mask)

    com_3d = (int(com[0]), int(com[1]), z_max_idx)
    return com_3d


def extract_square_proj_expand(image, single_mask, extra_pixels = 50):
    DAPI_stack, WGA_stack = image[:, 0, :, :], image[:, 2, :, :]

    _, _, comzi = nucleus_com(DAPI_stack, single_mask)  # Gets the nucleus stack of the middle of the cell

    sq_maski = square_mask(single_mask)
    props = regionprops(sq_maski.astype(int))
    if not props:
        min_row, min_col, max_row, max_col = 0, 0, sq_maski.shape[0], sq_maski.shape[1]
    else:
        min_row, min_col, max_row, max_col = props[0].bbox

    # Clip bbox to original stack bounds
    y_max, x_max = DAPI_stack.shape[1], DAPI_stack.shape[2]
    min_row = max(0, min_row)
    min_col = max(0, min_col)
    max_row = min(y_max, max_row)
    max_col = min(x_max, max_col)

    # Dimensions of the region of interest
    roi_height = max_row - min_row
    roi_width = max_col - min_col

    # Dimensions of the new canvas with extra space
    new_height = max(roi_height + 2 * extra_pixels, 1)
    new_width = max(roi_width + 2 * extra_pixels, 1)
 
    # Create new black canvas (filled with zeros)
    new_WGA_slice = np.zeros((new_height, new_width), dtype=WGA_stack.dtype)
    new_DAPI_slice = np.zeros((new_height, new_width), dtype=DAPI_stack.dtype)

    # Calculate the placement of the ROI in the new canvas
    new_min_row = extra_pixels 
    new_min_col = extra_pixels 
 
    # Extract the region of interest and place it in the center of the new canvas
    sq_WGA_slice = WGA_stack[comzi, min_row:max_row, min_col:max_col]
    # Ensure shapes align before assignment
    h_slice = min(roi_height, sq_WGA_slice.shape[0]) if sq_WGA_slice.ndim == 2 else 0
    w_slice = min(roi_width, sq_WGA_slice.shape[1]) if sq_WGA_slice.ndim == 2 else 0
    if h_slice > 0 and w_slice > 0:
        new_WGA_slice[new_min_row:new_min_row + h_slice, new_min_col:new_min_col + w_slice] = sq_WGA_slice[:h_slice, :w_slice]

    return new_WGA_slice, comzi

def remove_boundary(mask, buffer=50):
    if buffer <= 0:
        return mask
    h, w = mask.shape
    if buffer * 2 >= h or buffer * 2 >= w:
        # buffer too large -> return original mask (safe fallback)
        return mask.copy()
    return mask[buffer:-buffer, buffer:-buffer]

def closest_mask_2d(reference_mask, mask_array):
    ref_coords = np.argwhere(reference_mask)
    if len(ref_coords) == 0:
        return np.zeros_like(reference_mask)
    ref_center = np.mean(ref_coords, axis=0)
    labels = np.unique(mask_array)
    labels = labels[labels != 0]
    min_dist = float('inf')
    best_mask = np.zeros_like(reference_mask)
    for label in labels:
        candidate_mask = (mask_array == label)
        coords = np.argwhere(candidate_mask)
        center = np.mean(coords, axis=0)
        dist = np.linalg.norm(center - ref_center)
        if dist < min_dist:
            min_dist = dist
            best_mask = candidate_mask
    return best_mask.astype(np.uint8)

def get_traces(stacks, mask):
    # Return trace (z-profile) for the first channel in `stacks`
    for ch in stacks:
        ch_traces = []
        for z in ch:
            vals = z[mask > 0]
            if vals.size == 0:
                ch_traces.append(0.0)
            else:
                ch_traces.append(float(np.mean(vals)))
        return np.array(ch_traces)
    return np.array([])

def nuclei_centers_of_mass(stack, masks):
    ids = np.unique(masks)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return np.empty((0, 3))

    # If masks and stack have same dimensions use scipy directly
    try:
        if masks.ndim == stack.ndim:
            return np.array(center_of_mass(stack, labels=masks, index=ids))
    except Exception:
        pass

    # Common case: masks is 2D, stack is 3D -> get 2D centroid and pick z by max signal
    if masks.ndim == 2 and stack.ndim == 3:
        centers = []
        z_depth = stack.shape[0]
        for lab in ids:
            mask2d = (masks == lab)
            if not mask2d.any():
                continue
            com2d = center_of_mass(mask2d)
            z_sums = np.array([np.sum(stack[z][mask2d]) for z in range(z_depth)])
            if np.all(z_sums == 0):
                z_idx = int(z_depth // 2)
            else:
                z_idx = int(np.argmax(z_sums))
            centers.append((float(com2d[0]), float(com2d[1]), float(z_idx)))
        return np.array(centers)

    # Fallback to scipy (safe)
    try:
        return np.array(center_of_mass(stack, labels=masks, index=ids))
    except Exception:
        return np.empty((0, 3))

def remove_outliers_local(centers_of_mass, num_closest_points=20, z_threshold=2):
    if num_closest_points >= len(centers_of_mass):
        raise ValueError("num_closest_points must be less than the number of total points")
    filtered_data = []
    filtered_indices = []
    xs = np.array([coord[0] for coord in centers_of_mass])
    ys = np.array([coord[1] for coord in centers_of_mass])
    zs = np.array([coord[2] for coord in centers_of_mass])
    for i, (x, y, z) in enumerate(centers_of_mass):
        distances = np.sqrt((xs - x)**2 + (ys - y)**2 + (zs - z)**2)
        closest_indices = distances.argsort()[1:num_closest_points+1]
        z_closest = zs[closest_indices]
        mean_z = np.mean(z_closest)
        std_dev_z = np.std(z_closest)
        if abs(z - mean_z) <= z_threshold * std_dev_z:
            filtered_data.append((x, y, z))
            filtered_indices.append(i)
    return filtered_data, filtered_indices

def organize_data(mask_id, z_sep, stack_depth, metadata_row, filename):
    x_vals = ",".join(map(str, np.array(range(stack_depth)) * z_sep))

    return pd.DataFrame({
        "mask_id": [mask_id],
        "Slice_Seperation": z_sep,
        "X_vals": [x_vals],
        "file_name": [filename],
        "DJID": [metadata_row.get("djid", "")],
        "Eye": [metadata_row.get("eye", "")],
        "Treatment": [metadata_row.get("treatment", "")],
        "Stain": [metadata_row.get("stain", "")],
        "Time_Min": [metadata_row.get("time_min", "")],
        "eGFP_Value": [False],
        "eGFP_Raw_Intensity": [0.0],
        "in_rip": [False]
    })

def normalize(array):
    array = np.array(array)
    return (array - array.min()) / (array.max() - array.min())

def segment_images():
    """
    Step 1: Segment all images and save segmentation results.
    Creates a directory with mask images for each analyzed stack.
    """
    if GUI_helpers.metadata_df.empty:
        dpg.set_value("status_text", "No files are ready for segmentation.")
        return

    dpg.configure_item("segment_images_button", enabled=False)
    dpg.set_value("trace_file_status", "File: Starting segmentation...")
    dpg.set_value("trace_status_text", "Status: Preparing models...")

    # Create segmentation output directory
    folder_name = os.path.basename(GUI_helpers.current_folder.rstrip("/\\"))
    segmentation_dir = os.path.join(GUI_helpers.current_folder, f"{folder_name}_segmentation")
    os.makedirs(segmentation_dir, exist_ok=True)

    dapi_model_path = os.path.join(ROOT_DIR, 'CP_models', 'T5_DAPI_V4')
    dapi_model = denoise.CellposeDenoiseModel(gpu=True, model_type=dapi_model_path, restore_type="deblur_cyto3")

    model_path_wga = os.path.join(ROOT_DIR, 'CP_models', 'T5_WGA_V2')
    wga_model = models.CellposeModel(gpu=True, pretrained_model=model_path_wga)
    print('Done loading models for segmentation')

    for idx, row in GUI_helpers.metadata_df.iterrows():
        filename = row.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            dpg.set_value("trace_status_text", "Error: Invalid filename in metadata.")
            dpg.configure_item("segment_images_button", enabled=True)
            return

        file_path = os.path.join(GUI_helpers.current_folder, filename)
        if not os.path.exists(file_path):
            dpg.set_value("trace_status_text", f"Error: File not found - {file_path}")
            dpg.configure_item("segment_images_button", enabled=True)
            return

        z_min, z_max = int(row["z_min"]), int(row["z_max"])
        dpg.set_value("trace_file_status", f"File: {filename}")

        with nd2.ND2File(file_path) as f:
            stack = to_8bit(f.asarray())
            cropped_stack = stack[z_min:z_max+1]

        dapi_stack = cropped_stack[:, 0, :, :]
        proj = np.max(dapi_stack, axis=0)
        enhanced = auto_brightness_contrast(proj)

        print('Running dapi model for segmentation')
        dapi_masks, _, _, _ = dapi_model.eval(enhanced, diameter=None, channels=[0, 0])
        print('Done running dapi model')

        coords_3d = nuclei_centers_of_mass(dapi_stack, dapi_masks)
        filtered_coords, filtered_idxs = remove_outliers_local(coords_3d, num_closest_points=15, z_threshold=2)

        mask_ids = np.delete(np.unique(dapi_masks), 0) - 1

        # Save segmentation results for each stack
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(segmentation_dir, f"{base_name}_segmentation.npz")
        
        np.savez(output_file,
                 dapi_masks=dapi_masks,
                 filtered_idxs=filtered_idxs,
                 color_assignment=assign_colors_to_masks(dapi_masks),
                 stack=stack,
                 cropped_stack=cropped_stack,
                 dapi_stack=dapi_stack,
                 filename=filename,
                 z_min=z_min,
                 z_max=z_max)
        
        print(f"Saved segmentation to: {output_file}")
        # Verify color assignment was saved
        saved_data = np.load(output_file, allow_pickle=True)
        if 'color_assignment' in saved_data:
            ca = saved_data['color_assignment'].item()
            print(f"[DEBUG] Verified: {len(ca)} colors saved in npz")
        
        # Assign colors to maximize visual differences between adjacent masks
        color_assignment = assign_colors_to_masks(dapi_masks)
        
        # Save a visualization of the segmentation with colors and labels
        vis_file = os.path.join(segmentation_dir, f"{base_name}_segmentation_vis.png")
        save_segmentation_visualization(dapi_masks, color_assignment, vis_file)

        dpg.set_value("trace_status_text", f"Segmented {filename}")

    dpg.set_value("trace_file_status", "File: Done with segmentation")
    dpg.set_value("trace_status_text", f"Status: Segmentation complete. Ready for trace extraction.")
    dpg.configure_item("segment_images_button", enabled=True)
    dpg.configure_item("extract_traces_button", enabled=True)
    
    # Auto-load and display the last segmented file
    if GUI_helpers.opened_file:
        GUI_helpers.load_segmentation_if_available(GUI_helpers.opened_file)

def extract_traces():
    """
    Step 2: Extract traces from previously segmented images.
    Requires segment_images() to have been run first.
    """
    if GUI_helpers.metadata_df.empty:
        dpg.set_value("status_text", "No files are ready for analysis.")
        return

    global trace_data_df
    print('extract traces')

    dpg.configure_item("extract_traces_button", enabled=False)
    dpg.set_value("trace_file_status", "File: Starting trace extraction...")
    dpg.set_value("trace_status_text", "Status: Preparing...")

    # Load segmentation directory
    folder_name = os.path.basename(GUI_helpers.current_folder.rstrip("/\\"))
    segmentation_dir = os.path.join(GUI_helpers.current_folder, f"{folder_name}_segmentation")
    
    if not os.path.exists(segmentation_dir):
        dpg.set_value("trace_status_text", "Error: Segmentation not found. Run segmentation first.")
        dpg.configure_item("extract_traces_button", enabled=True)
        return

    results = []

    dapi_model_path = os.path.join(ROOT_DIR, 'CP_models', 'T5_DAPI_V4')
    dapi_model = denoise.CellposeDenoiseModel(gpu=True, model_type=dapi_model_path, restore_type="deblur_cyto3")

    model_path_wga = os.path.join(ROOT_DIR, 'CP_models', 'T5_WGA_V2')
    wga_model = models.CellposeModel(gpu=True, pretrained_model=model_path_wga)
    print('done loading models')

    for idx, row in GUI_helpers.metadata_df.iterrows():
        filename = row.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            dpg.set_value("trace_status_text", "Error: Invalid filename in metadata.")
            dpg.configure_item("extract_traces_button", enabled=True)
            return

        file_path = os.path.join(GUI_helpers.current_folder, filename)
        if not os.path.exists(file_path):
            dpg.set_value("trace_status_text", f"Error: File not found - {file_path}")
            dpg.configure_item("extract_traces_button", enabled=True)
            return

        # Load segmentation results
        base_name = os.path.splitext(filename)[0]
        seg_file = os.path.join(segmentation_dir, f"{base_name}_segmentation.npz")
        
        if not os.path.exists(seg_file):
            dpg.set_value("trace_status_text", f"Error: Segmentation not found for {filename}. Run segmentation first.")
            dpg.configure_item("extract_traces_button", enabled=True)
            return
        
        seg_data = np.load(seg_file, allow_pickle=True)
        dapi_masks = seg_data['dapi_masks']
        filtered_idxs = seg_data['filtered_idxs']
        stack = seg_data['stack']
        cropped_stack = seg_data['cropped_stack']
        dapi_stack = seg_data['dapi_stack']
        z_min = int(seg_data['z_min'])
        z_max = int(seg_data['z_max'])

        with nd2.ND2File(file_path) as f:
            z_sep = f.voxel_size().z

        dpg.set_value("trace_file_status", f"File: {filename}")
        print('Found file', z_sep)

        proj = np.max(dapi_stack, axis=0)
        enhanced = auto_brightness_contrast(proj)

        mask_ids = np.delete(np.unique(dapi_masks), 0) - 1

        print('Found masks', mask_ids)

        for i in mask_ids:
            if i not in filtered_idxs:
                continue

            dpg.set_value("trace_status_text", f"Extracting mask {i} of {len(mask_ids)}")

            single_mask = extract_masks(dapi_masks, i, reset_mask_ids=False)
            print('single mask', np.unique(single_mask))
            diam = get_mask_diameter(single_mask)
            expansion = 50

            sq_stacks = get_sq_stacks(stack, single_mask)
            print('Passed sq_stacks', i, single_mask.shape)

            expanded_sq, z_level = extract_square_proj_expand(stack, single_mask, expansion)

            expanded_mask, _, _ = wga_model.eval(expanded_sq, diameter=diam, channels=[0, 0])
            cleaned_mask = remove_boundary(expanded_mask, expansion)

            if len(np.unique(cleaned_mask)) == 1:
                continue
            elif len(np.unique(cleaned_mask)) > 2:
                cleaned_mask = closest_mask_2d(single_mask, cleaned_mask)

            file_base = row["filename"] if pd.notnull(row["filename"]) else ""
            cell_data = organize_data(i, z_sep, stack.shape[0], row, file_base)

            for ch_idx, ch_name in zip(range(min(stack.shape[1], 4)), ['DAPI', 'eGFP', 'WGA', 'GLUT1']):
                trace = get_traces(np.expand_dims(sq_stacks[ch_idx], axis=0), cleaned_mask)
                cell_data[f"Y_vals_{ch_name}"] = [trace] * len(cell_data)
                if ch_name == 'eGFP':
                    eGFP_sum = np.sum(sq_stacks[1][z_level][cleaned_mask.astype(bool)])
                    cell_data['eGFP_Raw_Intensity'] = eGFP_sum / np.sum(cleaned_mask)

            rip_ids = row.get("rip_cells", [])
            cell_data["in_rip"] = [i in rip_ids]

            results.append(cell_data)

        dpg.set_value("trace_status_text", f"Finished {filename}")

    if results:
        trace_data_df = pd.concat(results, ignore_index=True)
        if "eGFP_Raw_Intensity" in trace_data_df:
            egfp_vals = trace_data_df["eGFP_Raw_Intensity"].values
            normalized_vals = normalize(egfp_vals)
            trace_data_df["eGFP_Value"] = normalized_vals > 0.2

        # Rename mask_id to Segmentation_Mask_ID to match visualization
        trace_data_df.rename(columns={"mask_id": "Segmentation_Mask_ID"}, inplace=True)

        # Define folder_name once at the top level
        folder_name = os.path.basename(GUI_helpers.current_folder.rstrip("/\\"))
        processed_path = os.path.join(GUI_helpers.current_folder, f"{folder_name}_processed.csv")

        if dpg.get_value("opt_save_metadata"):
            csv_path = os.path.join(GUI_helpers.current_folder, f"{folder_name}_raw.csv")
            trace_data_df.to_csv(csv_path, index=False)
            print(f"Saved traces to: {csv_path}")

            if GUI_helpers.metadata_df is not None:
                meta_csv_path = os.path.join(GUI_helpers.current_folder, f"{folder_name}_metadata.csv")
                GUI_helpers.metadata_df.to_csv(meta_csv_path, index=False)
                print(f"Saved metadata to: {meta_csv_path}")
        
        # Save processed analysis
        if dpg.get_value("opt_save_analyzed"):
            processed_df = run_integral_analysis(trace_data_df)

            ## Post processing 
            drop_cols = ['X_vals', 'Y_vals_DAPI', 'Y_vals_eGFP','Y_vals_WGA', 'Y_vals_GLUT1',
                        'Cell','WGA_Middle_Indices', 'DAPI_peak_index',
                        'WGA_Top_Indices','WGA_Bottom_Indices',]

            rename_cols = {'Treatment':'Experimental_Condition', 'in_rip':'In_Rip',
                            'Time_Min': 'Time_Condition', 'Length':'Length_um'}

            processed_df.drop(columns= drop_cols, axis = 1, inplace = True)
            processed_df.rename(columns =  rename_cols, inplace = True)
            ##

            processed_df.to_csv(processed_path, index=False)
            print(f"Saved processed analysis to: {processed_path}")
            
    dpg.set_value("trace_file_status", "File: Done")
    dpg.set_value("trace_status_text", f"Status: Saved to {processed_path}")
    dpg.configure_item("extract_traces_button", enabled=True)
    
    # Save filtered segmentation visualization (cells removed shown in gray)
    folder_name = os.path.basename(GUI_helpers.current_folder.rstrip("/\\"))
    segmentation_dir = os.path.join(GUI_helpers.current_folder, f"{folder_name}_segmentation")
    
    for idx, row in GUI_helpers.metadata_df.iterrows():
        filename = row.get("filename")
        if isinstance(filename, str) and filename.strip():
            base_name = os.path.splitext(filename)[0]
            seg_file = os.path.join(segmentation_dir, f"{base_name}_segmentation.npz")
            if os.path.exists(seg_file):
                try:
                    seg_data = np.load(seg_file, allow_pickle=True)
                    dapi_masks = seg_data['dapi_masks']
                    filtered_idxs = seg_data['filtered_idxs']
                    color_assignment = assign_colors_to_masks(dapi_masks)
                    filtered_vis_file = os.path.join(segmentation_dir, f"{base_name}_segmentation_filtered_vis.png")
                    create_filtered_segmentation_visualization(dapi_masks, color_assignment, filtered_idxs, filtered_vis_file)
                except Exception as e:
                    print(f"Error saving filtered segmentation for {filename}: {e}")

def run_integral_analysis(trace_data_df):
    df = trace_data_df.copy()

    print(f"[DEBUG] Starting analysis with {len(df)} rows")

    # Add separation and cell identity
    df["Cell"] = df["file_name"].astype(str) + "_mask" + df["Segmentation_Mask_ID"].astype(str)

    # Peak detection
    df = WGA_Peaks_Finder_V2(df)
    print(f"[DEBUG] After WGA_Peaks_Finder_V2: {len(df)} rows")
    print("[DEBUG] Sample DAPI_peak_index values:")
    print(df["DAPI_peak_index"].head(10))
    print(df["DAPI_peak_index"].apply(type).value_counts())


    df = filter_out_unclear_DAPI(df)
    print(f"[DEBUG] After filter_out_unclear_DAPI: {len(df)} rows")

    if len(df) == 0:
        print("[ERROR] No valid cells remaining after DAPI filtering.")
        return df

    # Integrals
    df = Top_Bottom_Indices_V2(df)
    df = TopMidBot_Integrals_V2(df)
    df = Surface_Integrals_V2(df)

    df = Replace_NaNs_With_None(df)

    print(f"[DEBUG] Final dataframe shape: {df.shape}")
    return df

def WGA_Peaks_Finder_V2(dataframe, prom_val: float = 1.0):
    """
    Identifies WGA peaks before and after a single DAPI peak for each row.
    Adds:
    - WGA_Middle_Indices: [peak_before_dapi, peak_after_dapi]
    - DAPI_peak_index: index of peak in DAPI channel
    - Length: distance between WGA peaks in microns
    - Cell: integer ID
    """
    wga_middle = []
    dapi_peaks = []
    lengths = []
    cell_ids = []

    for idx, row in dataframe.iterrows():
        y_wga = row.get("Y_vals_WGA", [])
        y_dapi = row.get("Y_vals_DAPI", [])
        sep = row.get("Slice_Seperation", np.nan)

        print(f"[DEBUG] Cell index: {idx}")
        print(f"[DEBUG] y_wga type: {type(y_wga)}, len: {len(y_wga) if hasattr(y_wga, '__len__') else 'N/A'}")
        print(f"[DEBUG] y_dapi type: {type(y_dapi)}, len: {len(y_dapi) if hasattr(y_dapi, '__len__') else 'N/A'}")
        print(f"[DEBUG] sep: {sep}")

        dapi_dist = int(12 / sep)
        wga_dist = int(1.05 / sep)

        dapi_indices, _ = find_peaks(y_dapi, prominence=prom_val, distance=dapi_dist)
        print('DEBUG', dapi_indices)
        wga_indices, _ = find_peaks(y_wga, prominence=prom_val, distance=wga_dist)

        peak_before = np.nan
        peak_after = np.nan

        if len(dapi_indices) == 1:
            dapi_idx = dapi_indices[0]
            for peak in wga_indices:
                if peak < dapi_idx:
                    peak_before = peak
                elif peak > dapi_idx and np.isnan(peak_after):
                    peak_after = peak
                    break
        else:
            dapi_idx = np.nan

        dist = (peak_after - peak_before) * sep if not np.isnan(peak_before) and not np.isnan(peak_after) else np.nan

        wga_middle.append([peak_before, peak_after])
        dapi_peaks.append(dapi_idx)
        lengths.append(dist)
        cell_ids.append(idx)

    dataframe["WGA_Middle_Indices"] = wga_middle
    dataframe["DAPI_peak_index"] = dapi_peaks
    dataframe["Length"] = lengths
    dataframe["Cell"] = cell_ids

    return dataframe

def filter_out_unclear_DAPI(dataframe):
    """
    Keeps rows where 'DAPI_peak_index' is a valid number (not NaN or None).
    Prints out the number and identities of filtered-out cells for debugging.
    """

    valid_rows = dataframe[dataframe["DAPI_peak_index"].apply(lambda x: pd.notna(x) and isinstance(x, (int, float)))].copy()
    filtered_out = dataframe[~dataframe.index.isin(valid_rows.index)]

    if not filtered_out.empty:
        print("Filtered out cells (no valid DAPI peak):", filtered_out["Cell"].unique().tolist())
    else:
        print("No cells were filtered out.")

    return valid_rows.reset_index(drop=True)

def Top_Bottom_Indices_V2(dataframe, microns_extension: float = 1.5):
    '''
    Calculates WGA_Top_Indices and WGA_Bottom_Indices based on Slice_Seperation and WGA_Middle_Indices.
    '''
    grouped = dataframe.groupby('Cell')
    slice_separation = grouped['Slice_Seperation'].first()
    first_peaks = grouped['WGA_Middle_Indices'].apply(lambda x: x.iloc[0] if len(x) > 0 else [np.nan, np.nan])

    index_offset = (microns_extension / slice_separation).fillna(0).astype(int)

    l_middle = first_peaks.apply(lambda x: x[0] if len(x) > 0 else np.nan)
    r_middle = first_peaks.apply(lambda x: x[1] if len(x) > 1 else np.nan)

    l_top = np.maximum(l_middle - index_offset, 0)
    r_bot = r_middle + index_offset

    r_middle = r_middle.apply(lambda x: None if pd.isna(x) else x)
    r_bot = r_bot.apply(lambda x: None if pd.isna(x) else x)

    idx_df = pd.DataFrame({
        'Cell': grouped.size().index,
        'WGA_Top_Indices': list(zip(l_top, l_middle)),
        'WGA_Bottom_Indices': list(zip(r_middle, r_bot))
    })

    dataframe["WGA_Top_Indices"] = list(zip(l_top, l_middle))
    dataframe["WGA_Bottom_Indices"] = list(zip(r_middle, r_bot))
    return dataframe

def TopMidBot_Integrals_V2(dataframe):
    """
    Calculates WGA Top, Middle, Bottom integrals using defined index pairs.
    Adds columns: WGA_Top_Integral, WGA_Middle_Integral, WGA_Bottom_Integral
    """
    def integral_calculator(y_vals, indices):
        if not isinstance(indices, (list, tuple)) or pd.isna(indices[0]) or pd.isna(indices[1]):
            return None
        try:
            start_idx, end_idx = int(indices[0]), int(indices[1])
            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, len(y_vals))
            if start_idx >= end_idx:
                return None
            return float(np.sum(np.array(y_vals)[start_idx:end_idx]))
        except:
            return None

    for section in ['Middle', 'Top', 'Bottom']:
        WGA_col_name = f"WGA_{section}_Integral"
        WGA_index_col = f"WGA_{section}_Indices"
        dataframe[WGA_col_name] = dataframe.apply(lambda row: integral_calculator(row.get('Y_vals_WGA', []),
                                                                              row.get(WGA_index_col)), axis=1)
        GLUT1_col_name = f"GLUT1_{section}_Integral"
        dataframe[GLUT1_col_name] = dataframe.apply(lambda row: integral_calculator(row.get('Y_vals_GLUT1', []),
                                                                              row.get(WGA_index_col)), axis=1)
    return dataframe

def Surface_Integrals_V2(dataframe):
    def compute_surface(row):
        peak_indices = row.get("WGA_Middle_Indices", [np.nan, np.nan])
        x_vals = row.get("X_vals", [])
        y_G = row.get("Y_vals_GLUT1", [])
        y_W = row.get("Y_vals_WGA", [])
        sep = row.get("Slice_Seperation", None)
        idx_offset = int(1.5 / sep) if sep else 3

        def get_integral(idx, y_vals):
            if pd.isna(idx):
                return np.nan
            idx = int(idx)
            left = max(idx - idx_offset, 0)
            right = min(idx + idx_offset, len(x_vals))
            return np.sum(y_vals[left:right])

        top_G = get_integral(peak_indices[0], y_G)
        bot_G = get_integral(peak_indices[1], y_G)
        top_W = get_integral(peak_indices[0], y_W)
        bot_W = get_integral(peak_indices[1], y_W)

        return pd.Series({
            "GLUT1_Top_Surface_Integral": top_G,
            "GLUT1_Bot_Surface_Integral": bot_G,
            "WGA_Top_Surface_Integral": top_W,
            "WGA_Bot_Surface_Integral": bot_W,
            "Top_Surface_Ratio": top_G / top_W if not pd.isna(top_G) and not pd.isna(top_W) and top_W != 0 else np.nan,
            "Bot_Surface_Ratio": bot_G / bot_W if not pd.isna(bot_G) and not pd.isna(bot_W) and bot_W != 0 else np.nan,
        })
    
    surface_df = dataframe.apply(compute_surface, axis=1)
    for col in surface_df.columns:
        dataframe[col] = surface_df[col]
    return dataframe

def Replace_NaNs_With_None(dataframe):
    """
    Replaces all `NaN` values in a DataFrame with `None`, including those inside lists and tuples.
    """
    def replace_in_iterable(iterable):
        return type(iterable)(None if pd.isna(item) else item for item in iterable)

    def replace_nans(item):
        if isinstance(item, float) and np.isnan(item):
            return None
        elif isinstance(item, (int, str, list, np.ndarray)):
            return item
        elif pd.api.types.is_scalar(item) and pd.isna(item):
            return None
        return item

    return dataframe.applymap(replace_nans)
