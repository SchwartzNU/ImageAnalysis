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
        "Sex": [metadata_row.get("sex", "")],
        "Eye": [metadata_row.get("eye", "")],
        "Age": [metadata_row.get("age", "")],  # NEW
        "Genotype": [metadata_row.get("genotype", "")],  # NEW
        "Treatment": [metadata_row.get("treatment", "")],  # NEW
        "Time_Min": [metadata_row.get("time_min", "")],
        "eGFP_Value": [False],
        "eGFP_Raw_Intensity": [0.0],
        "in_rip": [False]
    })

def normalize(array):
    array = np.array(array)
    return (array - array.min()) / (array.max() - array.min())

def extract_traces():
    if GUI_helpers.metadata_df.empty:
        dpg.set_value("status_text", "No files are ready for analysis.")
        return

    global trace_data_df
    print('extract traces')

    dpg.configure_item("extract_traces_button", enabled=False)
    dpg.set_value("trace_file_status", "File: Starting...")
    dpg.set_value("trace_status_text", "Status: Preparing...")

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

        z_min, z_max = int(row["z_min"]), int(row["z_max"])
        dpg.set_value("trace_file_status", f"File: {filename}")

        with nd2.ND2File(file_path) as f:
            z_sep = f.voxel_size().z
            stack = to_8bit(f.asarray())
            cropped_stack = stack[z_min:z_max+1]
        print('Found file', z_sep)

        dapi_stack = cropped_stack[:, 0, :, :]
        proj = np.max(dapi_stack, axis=0)
        enhanced = auto_brightness_contrast(proj)

        print('Now running dapi model')
        dapi_masks, _, _, _ = dapi_model.eval(enhanced, diameter=None, channels=[0, 0])
        print('Done running dapi model')

        print('Starting coords')
        coords_3d = nuclei_centers_of_mass(dapi_stack, dapi_masks)
        print(len(coords_3d))
        print('Starting filtering')
        filtered_coords, filtered_idxs = remove_outliers_local(coords_3d, num_closest_points=15, z_threshold=2)

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

        trace_data_df["original_mask_id"] = trace_data_df["mask_id"]
        trace_data_df["mask_id"] = range(len(trace_data_df))

        if dpg.get_value("opt_save_metadata"):
            folder_name = os.path.basename(GUI_helpers.current_folder.rstrip("/\\"))
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
                        'original_mask_id', 'Cell','WGA_Middle_Indices', 'DAPI_peak_index',
                        'WGA_Top_Indices','WGA_Bottom_Indices',]

            rename_cols = {'Age': 'Age_Months', 'Treatment':'Experimental_Condition', 'in_rip':'In_Rip',
                            'Time_Min': 'Time_Condition', 'Length':'Length_um'}

            processed_df.drop(columns= drop_cols, axis = 1, inplace = True)
            processed_df.rename(columns =  rename_cols, inplace = True)
            ##

            processed_path = os.path.join(GUI_helpers.current_folder, f"{folder_name}_processed.csv")
            processed_df.to_csv(processed_path, index=False)
            print(f"Saved processed analysis to: {processed_path}")
            

    dpg.set_value("trace_file_status", "File: Done")
    dpg.set_value("trace_status_text", f"Status: Saved to {processed_path}")
    dpg.configure_item("extract_traces_button", enabled=True)

def run_integral_analysis(trace_data_df):
    df = trace_data_df.copy()

    print(f"[DEBUG] Starting analysis with {len(df)} rows")

    # Add separation and cell identity
    df["Cell"] = df["file_name"].astype(str) + "_mask" + df["mask_id"].astype(str)

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
