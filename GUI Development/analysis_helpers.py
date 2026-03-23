import os
import time
import numpy as np
import pandas as pd
import dearpygui.dearpygui as dpg
import nd2
import cv2
import warnings
from skimage import exposure, measure
from scipy.ndimage import center_of_mass, sum as nd_sum
from scipy.stats import skew
from scipy.signal import find_peaks, peak_widths
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels
from cellpose import models, denoise
import GUI_helpers

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

trace_data_df = None
wga_model_cache = None

def suppress_cellpose_torch_futurewarning():
    # Cellpose currently triggers a torch.load FutureWarning internally when
    # loading trusted local model files. Filter just that known third-party
    # warning so the GUI logs stay readable.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*weights_only=False.*",
    )

suppress_cellpose_torch_futurewarning()

def get_wga_model():
    global wga_model_cache
    if wga_model_cache is None:
        model_path_wga = os.path.join(ROOT_DIR, "CP_models", "T5_WGA_V2")
        wga_model_cache = models.CellposeModel(gpu=True, pretrained_model=model_path_wga)
    return wga_model_cache

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
    
    # Build a large palette with good spread in HSV.
    # OpenCV HSV: Hue 0-179, Saturation 0-255, Value 0-255.
    # We use golden-ratio hue stepping to avoid clustering and vary S/V to extend
    # beyond 180 unique hues when many masks are present.
    palette = []
    palette_size = max(72, len(unique_masks))
    hue_fracs = (np.arange(palette_size) * 0.61803398875) % 1.0
    sat_cycle = [0.95, 0.80, 0.65]
    val_cycle = [0.95, 0.85, 0.75]
    for i, hf in enumerate(hue_fracs):
        hue = int(hf * 179)
        sat = int(sat_cycle[i % len(sat_cycle)] * 255)
        val = int(val_cycle[(i // len(sat_cycle)) % len(val_cycle)] * 255)
        hsv_color = np.uint8([[[hue, sat, val]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
        r = float(rgb_color[0, 0, 0]) / 255.0
        g = float(rgb_color[0, 0, 1]) / 255.0
        b = float(rgb_color[0, 0, 2]) / 255.0
        palette.append((r, g, b))
    
    # Build adjacency graph from neighboring label pixels.
    # This avoids the previous O(n^2) pairwise mask-vs-mask dilation pass.
    adjacencies = defaultdict(set)
    h, w = mask_array.shape
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        if dy >= 0:
            a_y = slice(dy, h)
            b_y = slice(0, h - dy)
        else:
            a_y = slice(0, h + dy)
            b_y = slice(-dy, h)

        if dx >= 0:
            a_x = slice(dx, w)
            b_x = slice(0, w - dx)
        else:
            a_x = slice(0, w + dx)
            b_x = slice(-dx, w)

        a = mask_array[a_y, a_x]
        b = mask_array[b_y, b_x]
        valid = (a > 0) & (b > 0) & (a != b)
        if not np.any(valid):
            continue

        pairs = np.column_stack((a[valid].ravel(), b[valid].ravel()))
        pairs.sort(axis=1)
        for left, right in np.unique(pairs, axis=0):
            left = int(left)
            right = int(right)
            adjacencies[left].add(right)
            adjacencies[right].add(left)
    
    # Improved greedy assignment: prefer palette colors that maximize color-distance
    # to already-assigned neighboring colors so we use many distinct colors while
    # still avoiding exact adjacency matches.
    color_assignment = {}
    sorted_masks = sorted(unique_masks, key=lambda x: -len(adjacencies[x]))
    palette_usage = [0] * len(palette)
    palette_indices = {color: i for i, color in enumerate(palette)}

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
                # No assigned neighbors yet -> prefer colors that are not yet used globally.
                used_count = palette_usage[palette_indices[color]]
                score = 1.0 / (1 + used_count)
            else:
                # Score is the minimal squared distance to neighbors
                dists = [color_distance(color, ac) for ac in adjacent_colors]
                score = min(dists) if dists else 0.0

            if score > best_score:
                best_score = score
                best_color = color

        if best_color is None:
            # Fallback: pick a palette color with minimal global usage.
            best_idx = int(np.argmin(palette_usage))
            best_color = palette[best_idx]

        color_assignment[mask_id] = best_color
        palette_usage[palette_indices[best_color]] += 1

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

            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.putText(temp_mask, label_text, (x, y), font, font_scale, 255, thickness)

            # Blend text into RGBA using an explicit mask so black text is preserved.
            text_mask = temp_mask > 0
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

def create_labeled_segmentation_image(mask_array, color_assignment, filtered_idxs=None, label_min_area=5, force_labels=False, show_removed=False, output_size=None, label_filtered_only=False):
    """
    Create a segmentation visualization with colored, filled masks and numeric labels.
    Text color is chosen (black or white) based on background brightness.
    Returns: RGBA image as numpy array
    """
    mask_full = mask_array
    if output_size is not None:
        out_w, out_h = output_size
        if out_w <= 0 or out_h <= 0:
            raise ValueError("output_size must be positive (width, height)")
        # Nearest-neighbor to preserve mask ids for coloring.
        mask_array = cv2.resize(mask_array.astype(np.int32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)

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
    
    # Add labels with dynamic text color. LABEL ALL MASKS regardless of size.
    from skimage.measure import regionprops
    labels_added = 0
    labels_skipped = 0

    # Render labels on full resolution mask to avoid downscale blur,
    # then resize the text overlay if output_size is requested.
    full_h, full_w = mask_full.shape
    text_img_full = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    text_mask_full = np.zeros((full_h, full_w), dtype=np.uint8)

    filtered_set = set(filtered_idxs) if filtered_idxs is not None else None
    regions = regionprops(mask_full.astype(np.int32))
    scale_factor = 1.0
    if output_size is not None:
        scale_x = w / float(full_w) if full_w else 1.0
        scale_y = h / float(full_h) if full_h else 1.0
        scale_factor = min(scale_x, scale_y)

    for prop in regions:
        mask_id = prop.label
        if label_filtered_only and filtered_set is not None and (mask_id - 1) not in filtered_set:
            continue
        area = prop.area

        if not force_labels and area < label_min_area:
            labels_skipped += 1
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
        # Keep labels visible even for tiny masks
        font_scale = 0.6 if area >= 200 else 0.45
        if scale_factor < 1.0 and scale_factor > 0:
            # Compensate for downscale so text stays readable
            font_scale = min(font_scale / scale_factor, 2.0)
        thickness = 1

        text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_w, text_h = text_size

        # Center text on centroid and clamp to image bounds
        x = max(0, min(cx - text_w // 2, full_w - text_w))
        y = max(text_h, min(cy + text_h // 2, full_h))

        # Draw a mask for the text (always white) so black text is not lost
        temp_mask = np.zeros((full_h, full_w), dtype=np.uint8)
        cv2.putText(temp_mask, label_text, (x, y), font, font_scale, 255, thickness)
        if np.any(temp_mask):
            text_mask_full = np.maximum(text_mask_full, temp_mask)
            # Apply the requested text color (can be black or white)
            text_img_full[temp_mask > 0, 0] = text_color_bgr[0]
            text_img_full[temp_mask > 0, 1] = text_color_bgr[1]
            text_img_full[temp_mask > 0, 2] = text_color_bgr[2]
        labels_added += 1

    # Resize text overlay if needed and blend into RGBA
    text_img = text_img_full
    text_mask = text_mask_full
    if output_size is not None and (text_img_full.shape[1] != w or text_img_full.shape[0] != h):
        text_img = cv2.resize(text_img_full, (w, h), interpolation=cv2.INTER_NEAREST)
        text_mask = cv2.resize(text_mask_full, (w, h), interpolation=cv2.INTER_NEAREST)

    text_mask = text_mask > 0
    if np.any(text_mask):
        # text_img is BGR; convert to RGB channels
        rgba[text_mask, 0] = text_img[..., 2][text_mask] / 255.0
        rgba[text_mask, 1] = text_img[..., 1][text_mask] / 255.0
        rgba[text_mask, 2] = text_img[..., 0][text_mask] / 255.0
        rgba[text_mask, 3] = 1.0
    else:
        labels_skipped = len(regions)
    
    print(f"[DEBUG] Labels added: {labels_added}, skipped: {labels_skipped}")
    
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

def get_wga_target_diameter(mask):
    nucleus_diameter = float(get_mask_diameter(mask))
    return max(12.0, nucleus_diameter * 1.8)

def build_filtered_label_mask(dapi_masks, filtered_idxs):
    filtered_idxs = set(int(idx) for idx in filtered_idxs)
    keep_labels = [int(label_id) for label_id in np.unique(dapi_masks) if label_id > 0 and (int(label_id) - 1) in filtered_idxs]
    if not keep_labels:
        return np.zeros_like(dapi_masks, dtype=np.int32)
    return np.where(np.isin(dapi_masks, keep_labels), dapi_masks, 0).astype(np.int32)

def estimate_gfp_review_expansion(dapi_masks, filtered_idxs):
    diameters = []
    filtered_mask = build_filtered_label_mask(dapi_masks, filtered_idxs)
    for label_id in np.unique(filtered_mask):
        if label_id <= 0:
            continue
        diameters.append(get_mask_diameter(filtered_mask == label_id))
    if not diameters:
        return 4
    median_diameter = float(np.median(diameters))
    return max(2, min(6, int(round(median_diameter * 0.25))))

def dilate_binary_mask(mask, radius):
    mask_u8 = mask.astype(np.uint8)
    if radius <= 0:
        return mask_u8.astype(bool)
    kernel_size = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask_u8, kernel).astype(bool)

def score_gfp_region(gfp_stack, nucleus_mask):
    nucleus_mask = nucleus_mask.astype(bool)
    if not np.any(nucleus_mask):
        return np.nan

    fg_mask = dilate_binary_mask(nucleus_mask, radius=3)
    bg_inner = dilate_binary_mask(nucleus_mask, radius=5)
    bg_outer = dilate_binary_mask(nucleus_mask, radius=11)
    bg_mask = bg_outer & ~bg_inner

    slice_scores = []
    for z in range(gfp_stack.shape[0]):
        plane = gfp_stack[z]
        fg_vals = plane[fg_mask]
        if fg_vals.size == 0:
            continue
        bg_vals = plane[bg_mask]
        fg_signal = float(np.percentile(fg_vals, 90.0))
        bg_signal = float(np.median(bg_vals)) if bg_vals.size else 0.0
        slice_scores.append(fg_signal - bg_signal)

    if not slice_scores:
        return np.nan
    slice_scores = np.asarray(slice_scores, dtype=float)
    top_n = min(3, slice_scores.size)
    return float(np.mean(np.sort(slice_scores)[-top_n:]))

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

def get_sq_stacks(image, single_mask, channel_indices):
    min_row, min_col, max_row, max_col = get_square_mask_bbox(single_mask, image.shape[2], image.shape[3])

    sq_stacks = {
        "DAPI": image[:, channel_indices["dapi"], min_row:max_row, min_col:max_col],
        "WGA": image[:, channel_indices["wga"], min_row:max_row, min_col:max_col],
        "Stain": image[:, channel_indices["stain"], min_row:max_row, min_col:max_col],
    }
    if channel_indices.get("egfp") is not None:
        sq_stacks["eGFP"] = image[:, channel_indices["egfp"], min_row:max_row, min_col:max_col]

    return sq_stacks

def get_sq_stacks_from_bbox(image, bbox, channel_indices):
    min_row, min_col, max_row, max_col = bbox
    sq_stacks = {
        "DAPI": image[:, channel_indices["dapi"], min_row:max_row, min_col:max_col],
        "WGA": image[:, channel_indices["wga"], min_row:max_row, min_col:max_col],
        "Stain": image[:, channel_indices["stain"], min_row:max_row, min_col:max_col],
    }
    if channel_indices.get("egfp") is not None:
        sq_stacks["eGFP"] = image[:, channel_indices["egfp"], min_row:max_row, min_col:max_col]
    return sq_stacks

def extract_label_crop(label_image, label_id):
    ys, xs = np.where(label_image == int(label_id))
    if ys.size == 0 or xs.size == 0:
        return None, None
    min_row = int(ys.min())
    max_row = int(ys.max()) + 1
    min_col = int(xs.min())
    max_col = int(xs.max()) + 1
    cropped_mask = (label_image[min_row:max_row, min_col:max_col] == int(label_id)).astype(np.uint8)
    return cropped_mask, (min_row, min_col, max_row, max_col)

def get_square_mask_bbox(single_mask, y_max=None, x_max=None):
    sq_maski = square_mask(single_mask)
    props = regionprops(sq_maski.astype(int))
    if not props:
        min_row, min_col, max_row, max_col = 0, 0, sq_maski.shape[0], sq_maski.shape[1]
    else:
        min_row, min_col, max_row, max_col = props[0].bbox

    y_bound = sq_maski.shape[0] if y_max is None else y_max
    x_bound = sq_maski.shape[1] if x_max is None else x_max
    min_row = max(0, min_row)
    min_col = max(0, min_col)
    max_row = min(y_bound, max_row)
    max_col = min(x_bound, max_col)
    return min_row, min_col, max_row, max_col

def nucleus_com(single_channel, mask):
    masked_channel = single_channel * mask

    z_prof = np.sum(masked_channel, axis=(1, 2))
    z_max_idx = np.argmax(z_prof)

    com = center_of_mass(mask)

    com_3d = (int(com[0]), int(com[1]), z_max_idx)
    return com_3d


def extract_square_proj_expand(image, single_mask, channel_indices, extra_pixels = 50):
    DAPI_stack = image[:, channel_indices["dapi"], :, :]
    WGA_stack = image[:, channel_indices["wga"], :, :]

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

def extract_square_proj_expand_from_stacks(DAPI_stack, WGA_stack, single_mask, extra_pixels=50):
    _, _, comzi = nucleus_com(DAPI_stack, single_mask)

    sq_maski = square_mask(single_mask)
    props = regionprops(sq_maski.astype(int))
    if not props:
        min_row, min_col, max_row, max_col = 0, 0, sq_maski.shape[0], sq_maski.shape[1]
    else:
        min_row, min_col, max_row, max_col = props[0].bbox

    y_max, x_max = DAPI_stack.shape[1], DAPI_stack.shape[2]
    min_row = max(0, min_row)
    min_col = max(0, min_col)
    max_row = min(y_max, max_row)
    max_col = min(x_max, max_col)

    roi_height = max_row - min_row
    roi_width = max_col - min_col
    new_height = max(roi_height + 2 * extra_pixels, 1)
    new_width = max(roi_width + 2 * extra_pixels, 1)

    new_WGA_slice = np.zeros((new_height, new_width), dtype=WGA_stack.dtype)
    sq_WGA_slice = WGA_stack[comzi, min_row:max_row, min_col:max_col]
    h_slice = min(roi_height, sq_WGA_slice.shape[0]) if sq_WGA_slice.ndim == 2 else 0
    w_slice = min(roi_width, sq_WGA_slice.shape[1]) if sq_WGA_slice.ndim == 2 else 0
    if h_slice > 0 and w_slice > 0:
        new_WGA_slice[extra_pixels:extra_pixels + h_slice, extra_pixels:extra_pixels + w_slice] = sq_WGA_slice[:h_slice, :w_slice]

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

def get_peak_trace_index(stack, mask):
    trace = get_traces(np.expand_dims(stack, axis=0), mask)
    if trace.size == 0 or not np.any(np.isfinite(trace)):
        return None
    return int(np.nanargmax(trace))

def default_egfp_threshold(raw_intensities):
    vals = np.asarray(raw_intensities, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    low = float(np.min(vals))
    high = float(np.max(vals))
    if high <= low:
        return high
    return low + 0.2 * (high - low)

def prepare_gfp_review_data(file_path, segmentation_file, output_file, dapi_stack=None, wga_stack=None, egfp_stack=None):
    suppress_cellpose_torch_futurewarning()

    seg_data = np.load(segmentation_file, allow_pickle=True)
    dapi_masks = seg_data["dapi_masks"]
    filtered_idxs = set(int(idx) for idx in np.asarray(seg_data["filtered_idxs"]).tolist())
    z_min = int(seg_data["z_min"])
    z_max = int(seg_data["z_max"])

    if egfp_stack is not None:
        gfp_stack = np.asarray(egfp_stack)[z_min:z_max + 1]
    else:
        with nd2.ND2File(file_path) as f:
            stack = to_8bit(f.asarray())

        channel_indices = GUI_helpers.get_image_channel_indices(stack.shape[1])
        if channel_indices.get("egfp") is None:
            raise ValueError("Current file does not include a GFP channel.")

        gfp_stack = stack[z_min:z_max + 1, channel_indices["egfp"], :, :]

    gfp_proj = np.max(gfp_stack, axis=0)
    filtered_label_mask = build_filtered_label_mask(dapi_masks, filtered_idxs)
    expansion_distance = estimate_gfp_review_expansion(dapi_masks, filtered_idxs)
    overlay_masks = expand_labels(filtered_label_mask, distance=expansion_distance).astype(np.int32)
    mask_ids = []
    egfp_raw_intensities = []
    mask_labels = np.unique(filtered_label_mask)
    mask_labels = mask_labels[mask_labels > 0]
    for label_id in mask_labels:
        nucleus_mask = filtered_label_mask == int(label_id)
        if not np.any(nucleus_mask):
            continue
        raw_intensity = score_gfp_region(gfp_stack, nucleus_mask)
        mask_ids.append(int(label_id))
        egfp_raw_intensities.append(raw_intensity)

    threshold = default_egfp_threshold(egfp_raw_intensities)
    np.savez(
        output_file,
        gfp_proj=gfp_proj,
        overlay_masks=overlay_masks,
        mask_ids=np.asarray(mask_ids, dtype=np.int32),
        egfp_raw_intensities=np.asarray(egfp_raw_intensities, dtype=np.float32),
        default_threshold=np.float32(threshold),
        expansion_distance=np.int32(expansion_distance),
        review_version=np.int32(GUI_helpers.GFP_REVIEW_CACHE_VERSION),
        filename=os.path.basename(file_path),
    )
    return output_file

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
        com2d = np.array(center_of_mass(np.ones_like(masks, dtype=np.float32), labels=masks, index=ids))
        z_sums = np.vstack([nd_sum(stack[z], labels=masks, index=ids) for z in range(stack.shape[0])])
        z_idx = np.argmax(z_sums, axis=0)
        zero_signal = np.all(z_sums == 0, axis=0)
        z_idx[zero_signal] = stack.shape[0] // 2
        return np.column_stack((com2d[:, 0], com2d[:, 1], z_idx.astype(np.float32)))

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

def organize_data(mask_id, z_sep, stack_depth, metadata_row, filename, include_egfp=True):
    x_vals = ",".join(map(str, np.array(range(stack_depth)) * z_sep))

    data = {
        "mask_id": [mask_id],
        "Slice_Seperation": z_sep,
        "X_vals": [x_vals],
        "file_name": [filename],
        "DJID": [metadata_row.get("djid", "")],
        "Eye": [metadata_row.get("eye", "")],
        "Treatment": [metadata_row.get("treatment", "")],
        "Stain": [metadata_row.get("stain", "")],
        "Time_Min": [metadata_row.get("time_min", "")],
        "Segmented_Cell_Area_um2": [np.nan],
        "Segmented_Cell_Equivalent_Diameter_um": [np.nan],
        "Segmented_Cell_Roundness": [np.nan],
        "Stain_Middle_Mean": [np.nan],
        "in_rip": [False]
    }
    if include_egfp:
        data["eGFP_Value"] = [False]
        data["eGFP_Raw_Intensity"] = [0.0]
    return pd.DataFrame(data)

def normalize(array):
    array = np.array(array)
    return (array - array.min()) / (array.max() - array.min())

def compute_mask_roundness(mask):
    mask_uint8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_uint8):
        return np.nan

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.nan

    contour = max(contours, key=cv2.contourArea)
    area = float(np.count_nonzero(mask_uint8))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0 or area <= 0:
        return np.nan
    return float((4.0 * np.pi * area) / (perimeter ** 2))

def save_roundness_distribution_plot(dataframe, output_path):
    if "Segmented_Cell_Roundness" not in dataframe.columns:
        return

    roundness_vals = pd.to_numeric(dataframe["Segmented_Cell_Roundness"], errors="coerce").dropna()
    if roundness_vals.empty:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0.0, 1.05, 31)
    ax.hist(roundness_vals, bins=bins, color="#4C72B0", edgecolor="white")
    ax.set_title("Segmented Cell Roundness Distribution")
    ax.set_xlabel("Roundness (4πA / P²)")
    ax.set_ylabel("Cell count")
    ax.set_xlim(0.0, 1.05)

    median_val = float(np.median(roundness_vals))
    q1, q3 = np.percentile(roundness_vals, [25, 75])
    ax.axvline(median_val, color="#C44E52", linestyle="--", linewidth=1.5, label=f"Median = {median_val:.3f}")
    ax.axvline(q1, color="#55A868", linestyle=":", linewidth=1.2, label=f"Q1 = {q1:.3f}")
    ax.axvline(q3, color="#8172B3", linestyle=":", linewidth=1.2, label=f"Q3 = {q3:.3f}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def segment_images():
    """
    Step 1: Segment all images and save segmentation results.
    Creates a directory with mask images for each analyzed stack.
    """
    GUI_helpers.reconcile_metadata_with_folder(persist=True)
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

    suppress_cellpose_torch_futurewarning()

    recompute_existing = dpg.get_value("recompute_segmentation") if dpg.does_item_exist("recompute_segmentation") else False
    dapi_model = None
    metadata_changed = False

    for idx, row in GUI_helpers.metadata_df.iterrows():
        file_timer = time.perf_counter()
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

        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(segmentation_dir, f"{base_name}_segmentation.npz")
        gfp_review_file = os.path.join(segmentation_dir, f"{base_name}_gfp_review.npz")
        if os.path.exists(output_file) and not recompute_existing:
            dpg.set_value("trace_status_text", f"Loaded saved segmentation for {filename}")
            print(f"Using existing segmentation for {filename}: {output_file}")
            continue

        if dapi_model is None:
            dapi_model_path = os.path.join(ROOT_DIR, 'CP_models', 'T5_DAPI_V4')
            dapi_model = denoise.CellposeDenoiseModel(gpu=True, model_type=dapi_model_path, restore_type="deblur_cyto3")
            print('Done loading models for segmentation')

        with nd2.ND2File(file_path) as f:
            stack = to_8bit(f.asarray())
            cropped_stack = stack[z_min:z_max+1]
        channel_indices = GUI_helpers.get_image_channel_indices(stack.shape[1])

        dapi_stack = cropped_stack[:, 0, :, :]
        proj = np.max(dapi_stack, axis=0)
        enhanced = auto_brightness_contrast(proj)

        print(f'Running dapi model for segmentation: {filename}')
        step_timer = time.perf_counter()
        dapi_masks, _, _, _ = dapi_model.eval(enhanced, diameter=None, channels=[0, 0])
        print(f'Done running dapi model in {time.perf_counter() - step_timer:.2f}s')

        step_timer = time.perf_counter()
        coords_3d = nuclei_centers_of_mass(dapi_stack, dapi_masks)
        print(f'Computed nuclei centers in {time.perf_counter() - step_timer:.2f}s')

        step_timer = time.perf_counter()
        _, filtered_idxs = remove_outliers_local(coords_3d, num_closest_points=15, z_threshold=2)
        print(f'Filtered nuclei outliers in {time.perf_counter() - step_timer:.2f}s')

        step_timer = time.perf_counter()
        color_assignment = assign_colors_to_masks(dapi_masks)
        print(f'Assigned segmentation colors in {time.perf_counter() - step_timer:.2f}s')

        step_timer = time.perf_counter()
        np.savez(output_file,
                 dapi_masks=dapi_masks,
                 filtered_idxs=filtered_idxs,
                 color_assignment=color_assignment,
                 filename=filename,
                 z_min=z_min,
                 z_max=z_max)
        if os.path.exists(gfp_review_file):
            try:
                os.remove(gfp_review_file)
            except OSError:
                pass
        if channel_indices.get("egfp") is not None:
            metadata_changed |= GUI_helpers.invalidate_gfp_review_state(
                filename,
                remove_cache=False,
                clear_loaded=(GUI_helpers.opened_file == filename),
                persist=False,
            )
        print(f"Saved segmentation to: {output_file} in {time.perf_counter() - step_timer:.2f}s")

        # Save a visualization of the segmentation with colors and labels
        vis_file = os.path.join(segmentation_dir, f"{base_name}_segmentation_vis.png")
        step_timer = time.perf_counter()
        save_segmentation_visualization(dapi_masks, color_assignment, vis_file)
        print(f"Saved segmentation visualization in {time.perf_counter() - step_timer:.2f}s")

        dpg.set_value("trace_status_text", f"Segmented {filename}")
        print(f"Finished segmentation pipeline for {filename} in {time.perf_counter() - file_timer:.2f}s")

    dpg.set_value("trace_file_status", "File: Done with segmentation")
    if metadata_changed:
        GUI_helpers._persist_metadata()
        dpg.set_value("trace_status_text", "Status: Segmentation complete. Review GFP detection for GFP files before trace extraction.")
    else:
        dpg.set_value("trace_status_text", "Status: Segmentation complete. Ready for trace extraction.")
    dpg.configure_item("segment_images_button", enabled=True)
    dpg.configure_item("extract_traces_button", enabled=True)
    
    # Auto-load and display the last segmented file
    if GUI_helpers.opened_file:
        GUI_helpers.load_segmentation_if_available(GUI_helpers.opened_file)
        if GUI_helpers.gfp_channel_exists_for_opened_file():
            GUI_helpers.load_gfp_review_if_available(GUI_helpers.opened_file)
        else:
            GUI_helpers._clear_gfp_review_state()
            if dpg.does_item_exist("gfp_review_window"):
                dpg.hide_item("gfp_review_window")
            GUI_helpers.refresh_gfp_review_controls()
            GUI_helpers._update_gfp_status()

def extract_traces():
    """
    Step 2: Extract traces from previously segmented images.
    Requires segment_images() to have been run first.
    """
    GUI_helpers.reconcile_metadata_with_folder(persist=True)
    if GUI_helpers.metadata_df.empty:
        dpg.set_value("status_text", "No files are ready for analysis.")
        return

    global trace_data_df

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
    small_diameter_filtered = 0
    low_roundness_filtered = 0

    wga_model = None

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
        review_file = os.path.join(segmentation_dir, f"{base_name}_gfp_review.npz")
        
        if not os.path.exists(seg_file):
            dpg.set_value("trace_status_text", f"Error: Segmentation not found for {filename}. Run segmentation first.")
            dpg.configure_item("extract_traces_button", enabled=True)
            return
        
        seg_data = np.load(seg_file, allow_pickle=True)
        dapi_masks = seg_data['dapi_masks']
        filtered_idxs = seg_data['filtered_idxs']
        z_min = int(seg_data['z_min'])
        z_max = int(seg_data['z_max'])

        with nd2.ND2File(file_path) as f:
            voxel_size = f.voxel_size()
            z_sep = voxel_size.z
            x_sep = getattr(voxel_size, "x", np.nan)
            y_sep = getattr(voxel_size, "y", np.nan)
            stack = to_8bit(f.asarray())
        channel_indices = GUI_helpers.get_image_channel_indices(stack.shape[1])
        include_egfp = channel_indices.get("egfp") is not None
        egfp_threshold = row.get("egfp_threshold", np.nan)
        egfp_reviewed = row.get("egfp_reviewed", False)
        review_intensity_by_mask = {}
        review_overlay_masks = None
        if isinstance(egfp_reviewed, str):
            egfp_reviewed = egfp_reviewed.strip().lower() in {"1", "true", "yes"}
        if include_egfp and (not bool(egfp_reviewed) or not np.isfinite(pd.to_numeric(egfp_threshold, errors="coerce"))):
            dpg.set_value("trace_status_text", f"Error: Review GFP detection for {filename} before extracting traces.")
            dpg.configure_item("extract_traces_button", enabled=True)
            return
        if include_egfp:
            if not os.path.exists(review_file):
                dpg.set_value("trace_status_text", f"Error: GFP review data not found for {filename}. Prepare GFP review again.")
                dpg.configure_item("extract_traces_button", enabled=True)
                return
            review_data = np.load(review_file, allow_pickle=True)
            review_mask_ids = review_data["mask_ids"].astype(int).tolist()
            review_raw_vals = review_data["egfp_raw_intensities"].astype(float).tolist()
            review_intensity_by_mask = {int(mask_id): float(raw_val) for mask_id, raw_val in zip(review_mask_ids, review_raw_vals)}
            if "overlay_masks" in review_data:
                review_overlay_masks = review_data["overlay_masks"].astype(np.int32)
        dapi_stack = stack[z_min:z_max+1, channel_indices["dapi"], :, :]

        dpg.set_value("trace_file_status", f"File: {filename}")

        mask_ids = np.delete(np.unique(dapi_masks), 0) - 1

        filtered_mask_ids = [int(i) for i in mask_ids if int(i) in filtered_idxs]
        total_filtered_masks = len(filtered_mask_ids)
        for mask_counter, i in enumerate(filtered_mask_ids, start=1):
            if i not in filtered_idxs:
                continue

            if mask_counter == 1 or mask_counter == total_filtered_masks or (mask_counter % 10) == 0:
                dpg.set_value("trace_status_text", f"Extracting cell {mask_counter} of {total_filtered_masks}")

            label_id = int(i) + 1
            cleaned_mask = None
            sq_stacks = None
            z_level = None

            if review_overlay_masks is not None:
                cleaned_mask, bbox = extract_label_crop(review_overlay_masks, label_id)
                if cleaned_mask is not None and bbox is not None and np.any(cleaned_mask):
                    sq_stacks = get_sq_stacks_from_bbox(stack, bbox, channel_indices)

            if cleaned_mask is None or sq_stacks is None:
                single_mask = extract_masks(dapi_masks, i, reset_mask_ids=False)
                diam = get_wga_target_diameter(single_mask)
                expansion = 50
                min_row, min_col, max_row, max_col = get_square_mask_bbox(single_mask, stack.shape[2], stack.shape[3])
                reference_mask_crop = single_mask[min_row:max_row, min_col:max_col]
                sq_stacks = get_sq_stacks(stack, single_mask, channel_indices)
                expanded_sq, z_level = extract_square_proj_expand(stack, single_mask, channel_indices, expansion)
                if wga_model is None:
                    wga_model = get_wga_model()
                expanded_mask, _, _ = wga_model.eval(expanded_sq, diameter=diam, channels=[0, 0])
                cleaned_mask = remove_boundary(expanded_mask, expansion)

                if len(np.unique(cleaned_mask)) == 1:
                    continue
                elif len(np.unique(cleaned_mask)) > 2:
                    cleaned_mask = closest_mask_2d(reference_mask_crop, cleaned_mask)

            file_base = row["filename"] if pd.notnull(row["filename"]) else ""
            cell_data = organize_data(i, z_sep, stack.shape[0], row, file_base, include_egfp=include_egfp)
            cell_area_px = float(np.count_nonzero(cleaned_mask))
            pixel_area_um2 = float(x_sep) * float(y_sep) if np.isfinite(x_sep) and np.isfinite(y_sep) else np.nan
            cell_area_um2 = cell_area_px * pixel_area_um2 if np.isfinite(pixel_area_um2) else np.nan
            cell_eq_diameter_um = float(np.sqrt((4.0 * cell_area_um2) / np.pi)) if np.isfinite(cell_area_um2) and cell_area_um2 > 0 else np.nan
            if np.isfinite(cell_eq_diameter_um) and cell_eq_diameter_um < 5.0:
                small_diameter_filtered += 1
                continue
            cell_roundness = compute_mask_roundness(cleaned_mask)
            if np.isfinite(cell_roundness) and cell_roundness < 0.55:
                low_roundness_filtered += 1
                continue
            cell_data["Segmented_Cell_Area_um2"] = cell_area_um2
            cell_data["Segmented_Cell_Equivalent_Diameter_um"] = cell_eq_diameter_um
            cell_data["Segmented_Cell_Roundness"] = cell_roundness

            if cell_area_px > 0:
                largest_slice_idx = get_peak_trace_index(sq_stacks["WGA"], cleaned_mask)
                if largest_slice_idx is not None:
                    middle_slice = sq_stacks["Stain"][largest_slice_idx]
                    cell_data["Stain_Middle_Mean"] = float(np.mean(middle_slice[cleaned_mask.astype(bool)]))

            channel_order = ["DAPI"]
            if include_egfp:
                channel_order.append("eGFP")
            channel_order.extend(["WGA", "Stain"])

            for ch_name in channel_order:
                trace = get_traces(np.expand_dims(sq_stacks[ch_name], axis=0), cleaned_mask)
                cell_data[f"Y_vals_{ch_name}"] = [trace] * len(cell_data)
                if ch_name == 'eGFP':
                    egfp_raw_intensity = review_intensity_by_mask.get(int(i) + 1, np.nan)
                    if not np.isfinite(egfp_raw_intensity):
                        if z_level is None:
                            z_level = get_peak_trace_index(sq_stacks["eGFP"], cleaned_mask)
                            if z_level is None:
                                z_level = 0
                        eGFP_sum = np.sum(sq_stacks["eGFP"][z_level][cleaned_mask.astype(bool)])
                        egfp_raw_intensity = eGFP_sum / np.sum(cleaned_mask)
                    cell_data['eGFP_Raw_Intensity'] = egfp_raw_intensity
                    if np.isfinite(pd.to_numeric(egfp_threshold, errors="coerce")):
                        cell_data['eGFP_Value'] = bool(egfp_raw_intensity >= float(egfp_threshold))

            rip_ids = row.get("rip_cells", [])
            cell_data["in_rip"] = [i in rip_ids]

            results.append(cell_data)

        dpg.set_value("trace_status_text", f"Finished {filename}")

    if results:
        trace_data_df = pd.concat(results, ignore_index=True)

        # Rename mask_id to Segmentation_Mask_ID to match visualization
        trace_data_df.rename(columns={"mask_id": "Segmentation_Mask_ID"}, inplace=True)

        # Define folder_name once at the top level
        folder_name = os.path.basename(GUI_helpers.current_folder.rstrip("/\\"))
        processed_path = os.path.join(GUI_helpers.current_folder, f"{folder_name}_processed.csv")
        roundness_plot_path = os.path.join(GUI_helpers.current_folder, f"{folder_name}_roundness_distribution.png")
        save_roundness_distribution_plot(trace_data_df, roundness_plot_path)
        print(f"Saved roundness distribution plot to: {roundness_plot_path}")
        roundness_vals = pd.to_numeric(trace_data_df["Segmented_Cell_Roundness"], errors="coerce").dropna()
        if not roundness_vals.empty:
            q1, median, q3 = np.percentile(roundness_vals, [25, 50, 75])
            print(f"Roundness quartiles: Q1={q1:.3f}, median={median:.3f}, Q3={q3:.3f}")

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
            drop_cols = ['X_vals', 'Y_vals_DAPI', 'Y_vals_WGA', 'Y_vals_Stain',
                        'Stain_Mean_Intensity',
                        'Cell','WGA_Middle_Indices', 'DAPI_peak_index',
                        'WGA_Top_Indices','WGA_Bottom_Indices',]
            if 'Y_vals_eGFP' in processed_df.columns:
                drop_cols.append('Y_vals_eGFP')

            rename_cols = {'Treatment':'Experimental_Condition', 'in_rip':'In_Rip',
                            'Time_Min': 'Time_Condition', 'Length':'Length_um'}

            processed_df.drop(columns=[c for c in drop_cols if c in processed_df.columns], axis = 1, inplace = True)
            processed_df.rename(columns =  rename_cols, inplace = True)
            ##

            processed_df.to_csv(processed_path, index=False)
            print(f"Saved processed analysis to: {processed_path}")
            
    dpg.set_value("trace_file_status", "File: Done")
    dpg.set_value("trace_status_text", f"Status: Saved to {processed_path}")
    if small_diameter_filtered:
        print(f"Filtered out {small_diameter_filtered} cells with equivalent diameter < 5 um")
    if low_roundness_filtered:
        print(f"Filtered out {low_roundness_filtered} cells with roundness < 0.55")
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

    if "Stain_Middle_Mean" not in df.columns:
        df["Stain_Middle_Mean"] = np.nan
    else:
        missing_middle = pd.to_numeric(df["Stain_Middle_Mean"], errors="coerce").isna()
        df.loc[missing_middle, "Stain_Middle_Mean"] = np.nan

    def infer_stain_middle_from_traces(row):
        y_stain = row.get("Y_vals_Stain", [])
        y_wga = row.get("Y_vals_WGA", [])
        try:
            y_stain = np.asarray(y_stain, dtype=float)
            y_wga = np.asarray(y_wga, dtype=float)
        except Exception:
            return np.nan
        if y_stain.size == 0 or y_wga.size == 0:
            return np.nan
        if y_stain.size != y_wga.size or not np.any(np.isfinite(y_wga)):
            return np.nan
        idx = int(np.nanargmax(y_wga))
        if idx < 0 or idx >= y_stain.size:
            return np.nan
        return float(y_stain[idx])

    missing_middle = pd.to_numeric(df["Stain_Middle_Mean"], errors="coerce").isna()
    if missing_middle.any():
        df.loc[missing_middle, "Stain_Middle_Mean"] = df.loc[missing_middle].apply(infer_stain_middle_from_traces, axis=1)

    # Add separation and cell identity
    df["Cell"] = df["file_name"].astype(str) + "_mask" + df["Segmentation_Mask_ID"].astype(str)

    # Peak detection
    df = WGA_Peaks_Finder_V2(df)

    df = filter_out_unclear_DAPI(df)

    if len(df) == 0:
        print("[ERROR] No valid cells remaining after DAPI filtering.")
        return df

    # Profile means
    df = Middle_Means_V2(df)
    df = Surface_Means_V2(df)

    df = Replace_NaNs_With_None(df)
    return df

def smooth_profile(y_vals, sep, window_um=0.4):
    y_vals = np.asarray(y_vals, dtype=float)
    if y_vals.size == 0 or not np.isfinite(sep) or sep <= 0 or window_um <= 0:
        return y_vals

    window_pts = max(int(round(window_um / sep)), 1)
    if window_pts % 2 == 0:
        window_pts += 1
    if window_pts <= 1:
        return y_vals

    kernel = np.ones(window_pts, dtype=float) / float(window_pts)
    return np.convolve(y_vals, kernel, mode="same")

def WGA_Peaks_Finder_V2(dataframe, prom_val: float = 1.0, wga_prom_val: float = 0.75, wga_smooth_um: float = 0.4):
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
        y_wga = np.asarray(row.get("Y_vals_WGA", []), dtype=float)
        y_dapi = np.asarray(row.get("Y_vals_DAPI", []), dtype=float)
        sep = row.get("Slice_Seperation", np.nan)
        if not np.isfinite(sep) or sep <= 0:
            wga_middle.append([np.nan, np.nan])
            dapi_peaks.append(np.nan)
            lengths.append(np.nan)
            cell_ids.append(idx)
            continue

        dapi_dist = max(int(12 / sep), 1)
        wga_dist = max(int(1.05 / sep), 1)

        dapi_indices, _ = find_peaks(y_dapi, prominence=prom_val, distance=dapi_dist)
        y_wga_smoothed = smooth_profile(y_wga, sep, window_um=wga_smooth_um)
        wga_indices, _ = find_peaks(y_wga_smoothed, prominence=wga_prom_val, distance=wga_dist)

        peak_before = np.nan
        peak_after = np.nan

        if len(dapi_indices) == 1:
            dapi_idx = dapi_indices[0]
            before_candidates = wga_indices[wga_indices < dapi_idx]
            after_candidates = wga_indices[wga_indices > dapi_idx]
            if len(before_candidates) > 0:
                peak_before = int(before_candidates[-1])
            if len(after_candidates) > 0:
                peak_after = int(after_candidates[0])
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

def Middle_Means_V2(dataframe):
    """
    Calculates the WGA mean for the middle region between the two WGA peaks.
    Stain_Middle_Mean is expected to be precomputed from the inferred largest cell slice.
    """
    def mean_calculator(y_vals, indices):
        if not isinstance(indices, (list, tuple)) or pd.isna(indices[0]) or pd.isna(indices[1]):
            return None
        try:
            start_idx, end_idx = int(indices[0]), int(indices[1])
            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, len(y_vals))
            if start_idx >= end_idx:
                return None
            segment = np.asarray(y_vals)[start_idx:end_idx]
            if segment.size == 0:
                return None
            return float(np.mean(segment))
        except Exception:
            return None

    dataframe["WGA_Middle_Mean"] = dataframe.apply(
        lambda row: mean_calculator(row.get("Y_vals_WGA", []), row.get("WGA_Middle_Indices")),
        axis=1
    )
    return dataframe

def Surface_Means_V2(dataframe):
    def compute_surface(row):
        peak_indices = row.get("WGA_Middle_Indices", [np.nan, np.nan])
        y_G = row.get("Y_vals_Stain", [])
        y_W = row.get("Y_vals_WGA", [])
        sep = row.get("Slice_Seperation", np.nan)
        y_W_arr = np.asarray(y_W, dtype=float)
        y_G_arr = np.asarray(y_G, dtype=float)
        y_W_smoothed = smooth_profile(y_W_arr, sep, window_um=0.4)

        def get_window_bounds(idx):
            if pd.isna(idx):
                return (None, None, np.nan)
            idx = int(idx)
            try:
                widths, _, left_ips, right_ips = peak_widths(y_W_smoothed, [idx], rel_height=0.5)
            except Exception:
                return (None, None, np.nan)
            if len(widths) == 0:
                return (None, None, np.nan)

            left = max(int(np.floor(left_ips[0])), 0)
            right = min(int(np.ceil(right_ips[0])), len(y_W_smoothed))
            if right <= left:
                right = min(left + 1, len(y_W_arr))
            width_um = float(widths[0] * sep) if pd.notna(sep) and sep != 0 else np.nan
            return (left, right, width_um)

        def get_mean(bounds, y_vals):
            left, right, _ = bounds
            if left is None or right is None:
                return np.nan
            segment = np.asarray(y_vals[left:right], dtype=float)
            if segment.size == 0:
                return np.nan
            return float(np.mean(segment))

        top_bounds = get_window_bounds(peak_indices[0])
        bot_bounds = get_window_bounds(peak_indices[1])
        top_G = get_mean(top_bounds, y_G_arr)
        bot_G = get_mean(bot_bounds, y_G_arr)
        top_W = get_mean(top_bounds, y_W_arr)
        bot_W = get_mean(bot_bounds, y_W_arr)
        mid_stain = row.get("Stain_Middle_Mean", np.nan)

        return pd.Series({
            "Stain_Top_Surface_Mean": top_G,
            "Stain_Bot_Surface_Mean": bot_G,
            "WGA_Top_Surface_Mean": top_W,
            "WGA_Bot_Surface_Mean": bot_W,
            "WGA_Top_Surface_Width_um": top_bounds[2],
            "WGA_Bot_Surface_Width_um": bot_bounds[2],
            "Top_Surface_Ratio": top_G / top_W if not pd.isna(top_G) and not pd.isna(top_W) and top_W != 0 else np.nan,
            "Bot_Surface_Ratio": bot_G / bot_W if not pd.isna(bot_G) and not pd.isna(bot_W) and bot_W != 0 else np.nan,
            "Stain_Top_Surface_To_Stain_Middle_Ratio": (
                top_G / mid_stain
                if not pd.isna(top_G) and not pd.isna(mid_stain) and mid_stain != 0
                else np.nan
            ),
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
