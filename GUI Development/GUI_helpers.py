import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import dearpygui.dearpygui as dpg
import nd2
from cellpose import models, denoise
from skimage.morphology import binary_erosion, disk
from skimage import img_as_ubyte, exposure
import cv2
from fdialog import FileDialog

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

current_folder = None
opened_file = None
channel_zstack = None
channel2_stack = None
gray_img = None
mask_array = None
colors = {}
selected_masks = []
segmentation_masks = None  # New: store segmentation masks
segmentation_colors = {}   # New: store assigned colors
segmentation_filtered_idxs = None  # New: store which masks pass filtering
metadata_df = pd.DataFrame(columns=["filename", "z_min", "z_max", "rip_cells", "eye", "time_min", "djid", "treatment", "stain"])
texture_cache = None
last_show_masks = True
last_selected = []
display_map = {}
confirmed_rip_masks = {}
display_map = {}
confirmed_rip_masks = {}


# Texture is created by GUI_Imaging.py at startup; we just use it


def to_8bit(arr):
    norm = arr.astype(np.float32)
    if norm.size and norm.max() > 0:
        norm /= norm.max()
    else:
        norm = norm * 0.0
    return img_as_ubyte(norm)

def _set_dynamic_texture_from_array(rgba, texture_tag="dynamic_texture"):
    """
    Update a DearPyGui texture by resizing to fit the display area.
    Main image: scales to fit 1024x512
    """
    if rgba is None:
        return
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("rgba must be HxWx4")

    h, w = arr.shape[0], arr.shape[1]
    print(f"[DEBUG] _set_dynamic_texture_from_array: input shape {h}x{w}, texture={texture_tag}")

    # For now, only handle main image texture
    # Segmentation is handled directly in display_segmentation_filtered
    if texture_tag != "dynamic_texture":
        print(f"[WARN] _set_dynamic_texture_from_array called with unexpected texture_tag: {texture_tag}")
        return

    # Main image: scale to fit 1024x512 maintaining aspect ratio
    # Calculate scale to fit within 1024x512
    scale = min(1024 / w, 512 / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    print(f"[DEBUG] Scaling main image by {scale:.3f}: {new_w}x{new_h}")
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad to 1024x512
    canvas = np.zeros((512, 1024, 4), dtype=np.float32)
    y_offset = (512 - new_h) // 2
    x_offset = (1024 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Flatten and update texture
    flat = canvas.flatten().tolist()
    try:
        dpg.set_value(texture_tag, flat)
        print(f"[DEBUG] Texture updated successfully")
    except Exception as exc:
        print(f"[WARN] set_value failed: {exc}")

def normalize_image_fixed(img):
    return img.astype(np.float32) / 255.0

def auto_brightness_contrast(image):
    normalized_image = image.astype(np.float32) / 255.0
    equalized_image = exposure.equalize_adapthist(normalized_image)
    equalized_image = (equalized_image * 255).astype(np.uint8)
    return equalized_image

def max_proj(channel_zstack):
    return np.max(channel_zstack, axis=0)

def draw_mask_outlines():
    print("\n[DEBUG] draw_mask_outlines called")
    global mask_array, gray_img, colors, selected_masks
    if mask_array is None or gray_img is None:
        return
    if not dpg.get_value("show_masks_checkbox"):
        return

    outline_rgba = np.zeros((*gray_img.shape, 4), dtype=np.float32)
    for m in np.unique(mask_array):
        if m == 0:
            continue
        mask = (mask_array == m).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = colors.get(m, np.random.rand(3))
        for contour in contours:
            cv2.polylines(outline_rgba, [contour], isClosed=True, color=(*col, 1.0), thickness=1)
            if m in selected_masks:
                cv2.fillPoly(outline_rgba, [contour], color=(*col, 0.4))

    base = to_8bit(gray_img).astype(np.float32) / 255.0
    rgba = np.zeros((*gray_img.shape, 4), dtype=np.float32)
    rgba[..., :3] = base[..., None]
    rgba[..., 3] = 1.0

    mask_alpha = outline_rgba[..., 3:4]
    rgba[..., :3] = (1 - mask_alpha) * rgba[..., :3] + mask_alpha * outline_rgba[..., :3]
    rgba[..., 3] = np.clip(rgba[..., 3] + outline_rgba[..., 3], 0, 1)

    _set_dynamic_texture_from_array(rgba)

def blend_with_masks(gray):
    h, w = gray.shape
    base = to_8bit(gray).astype(np.float32) / 255.0
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = base[..., None]
    rgba[..., 3] = 1.0
    return rgba

def refresh_contents_list(sender=None, app_data=None, user_data=None):
    global current_folder, display_map
    display_map = {}

    if current_folder is None:
        dpg.configure_item("contents_list", items=[])
        return

    files = sorted([f for f in os.listdir(current_folder) if f.lower().endswith('.nd2')])
    display_items = []

    for f in files:
        tags = []
        if f in metadata_df["filename"].values:
            row = metadata_df.loc[metadata_df["filename"] == f].iloc[0]
            if isinstance(row["rip_cells"], list) and row["rip_cells"]:
                tags.append("[RIP]")
            tags.append("[SAVED]")
        tag_string = ''.join(tags)
        display = f"{tag_string} {f}" if tag_string else f
        display_items.append(display)
        display_map[display] = f

    dpg.configure_item("contents_list", items=display_items)

def handle_folder_selection(folder):
    global current_folder, opened_file
    current_folder = folder
    opened_file = None
    two_level = f"{os.path.basename(os.path.dirname(folder))}/{os.path.basename(folder)}"
    dpg.set_value("dir_path_repeat", two_level)
    refresh_contents_list()
    for tag in ("z_range_group", "rip_group", "wga_group"):
        if dpg.does_item_exist(tag):
            dpg.hide_item(tag)
    dpg.set_value("status_text", "Folder loaded")

def get_folder_picker():
    def folder_selected_callback(paths):
        if paths:
            handle_folder_selection(paths[0])
    return FileDialog(
        callback=folder_selected_callback,
        dirs_only=True,
        default_path=".",
        modal=False,
        allow_drag=False
    )   

def open_folder_dialog(sender, app_data, user_data):
    global current_folder, opened_file
    root = tk.Tk(); root.withdraw()
    folder = filedialog.askdirectory(); root.destroy()
    if not folder:
        return
    current_folder = folder
    opened_file = None
    two_level = f"{os.path.basename(os.path.dirname(folder))}\\{os.path.basename(folder)}"
    dpg.set_value("dir_path_repeat", two_level)
    refresh_contents_list()
    for tag in ("z_range_group","rip_group","wga_group"):
        if dpg.does_item_exist(tag):
            dpg.hide_item(tag)
    dpg.set_value("status_text", "Folder loaded")

def load_segmentation_if_available(filename):
    """
    Load segmentation masks and assign colors if available.
    """
    global segmentation_masks, segmentation_colors, segmentation_filtered_idxs
    
    # Clear any previous segmentation display immediately
    if dpg.does_item_exist("segmentation_window"):
        try:
            dpg.hide_item("segmentation_window")
        except Exception:
            pass
    # Clear segmentation texture to blank
    try:
        if dpg.does_item_exist("segmentation_texture"):
            blank = np.zeros((512, 1024, 4), dtype=np.float32).flatten().tolist()
            dpg.set_value("segmentation_texture", blank)
    except Exception:
        pass
    # Do not clear the main image texture here; keep the image visible while loading segmentation

    if current_folder is None or filename is None:
        segmentation_masks = None
        segmentation_colors = {}
        segmentation_filtered_idxs = None
        return
    
    folder_name = os.path.basename(current_folder.rstrip("/\\"))
    segmentation_dir = os.path.join(current_folder, f"{folder_name}_segmentation")
    
    base_name = os.path.splitext(filename)[0]
    seg_file = os.path.join(segmentation_dir, f"{base_name}_segmentation.npz")
    
    # Remove all previous segmentation artifacts on load so old visualizations don't auto-load
    if os.path.exists(segmentation_dir):
        for f in os.listdir(segmentation_dir):
            try:
                full = os.path.join(segmentation_dir, f)
                # Only remove files (not directories)
                if os.path.isfile(full):
                    os.remove(full)
            except Exception:
                pass
    
    if os.path.exists(seg_file):
        try:
            import analysis_helpers
            seg_data = np.load(seg_file, allow_pickle=True)
            segmentation_masks = seg_data['dapi_masks']
            segmentation_filtered_idxs = seg_data['filtered_idxs']
            # Always recompute a fresh color assignment to ensure good color diversity
            segmentation_colors = analysis_helpers.assign_colors_to_masks(segmentation_masks)
            print(f"[DEBUG] Recomputed {len(segmentation_colors)} colors for visualization")
            # If the file contained a saved color_assignment, keep it in memory for auditing
            if 'color_assignment' in seg_data:
                try:
                    color_dict = seg_data['color_assignment'].item()
                    print(f"[DEBUG] Found saved color_assignment in npz with {len(color_dict)} entries (not used for display)")
                except Exception:
                    pass
            print(f"Loaded segmentation for {filename}")
            # Automatically display the segmentation with filtered masks only
            display_segmentation_filtered()
        except Exception as e:
            print(f"Error loading segmentation: {e}")
            segmentation_masks = None
            segmentation_colors = {}
            segmentation_filtered_idxs = None
    else:
        segmentation_masks = None
        segmentation_colors = {}
        segmentation_filtered_idxs = None

def display_segmentation_filtered():
    """
    Display the loaded segmentation with colors and labels, showing only filtered masks.
    """
    global segmentation_masks, segmentation_colors, segmentation_filtered_idxs
    
    print(f"[DEBUG] display_segmentation_filtered called")
    print(f"[DEBUG] segmentation_masks is None: {segmentation_masks is None}")
    print(f"[DEBUG] segmentation_filtered_idxs: {segmentation_filtered_idxs}")
    
    if segmentation_masks is None:
        print("[DEBUG] No segmentation masks loaded, hiding window")
        if dpg.does_item_exist("segmentation_window"):
            dpg.hide_item("segmentation_window")
        return
    
    import analysis_helpers
    # We'll pass the original masks and filtered idxs to the viz function so it can
    # optionally render removed masks semi-transparently.
    filtered_masks = segmentation_masks
    
    print(f"[DEBUG] segmentation_colors has {len(segmentation_colors)} colors")
    if segmentation_colors:
        color_vals = list(segmentation_colors.values())
        print(f"[DEBUG] First 3 colors: {color_vals[:3]}")
        unique_colors = set(color_vals)
        print(f"[DEBUG] Unique colors: {len(unique_colors)}")
    
    # Read user preference for showing removed masks; labeling uses a small default area
    show_removed = dpg.get_value("show_removed_masks") if dpg.does_item_exist("show_removed_masks") else False

    seg_viz = analysis_helpers.create_labeled_segmentation_image(filtered_masks, segmentation_colors,
                                                                 filtered_idxs=segmentation_filtered_idxs,
                                                                 label_min_area=5,
                                                                 force_labels=False,
                                                                 show_removed=show_removed)
    h, w = seg_viz.shape[:2]
    print(f"[DEBUG] Segmentation viz shape: {h}x{w}")
    
    # Scale to 1024x512 for display (match main image)
    scale = min(1024 / w, 512 / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    seg_viz_resized = cv2.resize(seg_viz, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a canvas of the exact size needed (512 high to match main image)
    canvas = np.zeros((512, 1024, 4), dtype=np.float32)
    y_offset = (512 - new_h) // 2
    x_offset = (1024 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = seg_viz_resized
    
    # Update texture directly
    flat = canvas.flatten().tolist()
    try:
        if dpg.does_item_exist("segmentation_texture"):
            dpg.set_value("segmentation_texture", flat)
            print(f"[DEBUG] Set segmentation_texture value")
        if dpg.does_item_exist("segmentation_window"):
            dpg.show_item("segmentation_window")
            print(f"[DEBUG] Showed segmentation_window")
        print(f"[DEBUG] Displayed segmentation visualization")
    except Exception as e:
        print(f"[ERROR] updating segmentation texture: {e}")
        import traceback
        traceback.print_exc()

def display_segmentation():
    """
    Display the loaded segmentation with colors and labels.
    """
    global gray_img, segmentation_masks, segmentation_colors
    
    if segmentation_masks is None:
        return
    
    import analysis_helpers
    # Use default labeling behavior (no force labels, small min area)
    seg_viz = analysis_helpers.create_labeled_segmentation_image(segmentation_masks, segmentation_colors,
                                                                 filtered_idxs=None,
                                                                 label_min_area=5,
                                                                 force_labels=False,
                                                                 show_removed=False)
    h, w = seg_viz.shape[:2]
    
    # Scale to 1024x512 for display (match main image)
    scale = min(1024 / w, 512 / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    seg_viz_resized = cv2.resize(seg_viz, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a canvas of the exact size needed (512 high to match main image)
    canvas = np.zeros((512, 1024, 4), dtype=np.float32)
    y_offset = (512 - new_h) // 2
    x_offset = (1024 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = seg_viz_resized
    
    # Update texture directly
    flat = canvas.flatten().tolist()
    try:
        if dpg.does_item_exist("segmentation_texture"):
            dpg.set_value("segmentation_texture", flat)
    except Exception as e:
        print(f"[ERROR] updating segmentation texture: {e}")

def contents_list_callback(sender, app_data, user_data):
    display_name = dpg.get_value("contents_list")
    sel = display_map.get(display_name, display_name)
    if sel != opened_file:
        for tag in ["z_range_group", "z_min_slider", "z_max_slider", "rip_group", "wga_group", "wga_checkbox", "wga_slider", "identifiers_group"]:
            if dpg.does_item_exist(tag):
                dpg.hide_item(tag)
        dpg.set_value("status_text", f"Selected: {sel}")
    else:
        for tag in ["z_range_group", "z_min_slider", "z_max_slider", "wga_group", "wga_checkbox", "wga_slider", "identifiers_group"]:
            if dpg.does_item_exist(tag):
                dpg.show_item(tag)
        if opened_file in metadata_df['filename'].values:
            dpg.show_item("rip_group")

def open_nd2_callback(sender, app_data, user_data):
    global opened_file, channel_zstack, channel2_stack, gray_img, mask_array, colors, selected_masks, texture_cache
    display_name = dpg.get_value("contents_list")
    sel = display_map.get(display_name, display_name)
    if not sel:
        dpg.set_value("status_text", "No file selected")
        return

    if sel != opened_file:
        dpg.set_value("status_text", f"Loading: {sel}")

        # Reset internal state
        mask_array = None
        selected_masks.clear()
        colors.clear()
        texture_cache = None
        
        # Load segmentation if available
        load_segmentation_if_available(sel)

        # Load image data
        path = os.path.join(current_folder, sel)
        with nd2.ND2File(path) as f:
            stack8 = to_8bit(f.asarray())
        channel_zstack = stack8[:, 0, :, :]
        channel2_stack = stack8[:, 2, :, :]
        gray_img = max_proj(channel_zstack)
        opened_file = sel
        dpg.set_value("contents_list", sel)

        # Auto-fill DJID and Eye based on filename
        digits = ''.join(filter(str.isdigit, sel))
        djid_guess = digits[:4]
        dpg.set_value("djid_input", djid_guess)

        eye_guess = ""
        try:
            idx = sel.index(djid_guess) + len(djid_guess)
            if idx < len(sel):
                eye_char = sel[idx].upper()
                if eye_char in ["R", "L"]:
                    eye_guess = eye_char
        except (ValueError, IndexError):
            pass
        dpg.set_value("eye_combo", eye_guess)

        # Clear other identifier fields
        dpg.set_value("time_input", "")
        dpg.set_value("treatment_combo", "")
        dpg.set_value("stain_combo", "")

        # Reset WGA widgets
        dpg.set_value("wga_checkbox", False)
        dpg.configure_item("wga_slider", min_value=0, max_value=channel2_stack.shape[0] - 1)

        # Reset RIP state
        dpg.set_value("rip_checkbox", False)
        dpg.hide_item("run_rip_button")
        dpg.hide_item("show_masks_checkbox")
        dpg.set_value("show_masks_checkbox", False)
        dpg.hide_item("confirm_masks_button")
        dpg.hide_item("selected_mask_count")
        dpg.set_value("selected_mask_count", "Cells in rip: []")

        # Update texture view
        update_texture(gray_img, force=True)

        # Rebuild and show UI
        add_z_range_widget("contents_window", channel_zstack.shape[0])
        dpg.show_item("identifiers_group")
        dpg.show_item("wga_group")
        dpg.show_item("wga_checkbox")
        dpg.show_item("wga_slider")
        dpg.show_item("z_range_group")
        dpg.show_item("save_metadata_button")
        if sel in metadata_df["filename"].values:
            dpg.show_item("rip_group")

        dpg.set_value("status_text", f"Loaded: {sel}")

    else:
        dpg.set_value("status_text", f"Already loaded: {sel}")

    # Populate saved metadata if it exists
    if sel in metadata_df["filename"].values:
        row = metadata_df.loc[metadata_df["filename"] == sel].iloc[0]
        dpg.set_value("z_min_slider", int(row["z_min"]))
        dpg.set_value("z_max_slider", int(row["z_max"]))
        if pd.notnull(row["eye"]):
            dpg.set_value("eye_combo", row["eye"])
        if pd.notnull(row["time_min"]):
            dpg.set_value("time_input", str(int(row["time_min"])))
        if pd.notnull(row["djid"]):
            dpg.set_value("djid_input", str(row["djid"]))
        if pd.notnull(row["treatment"]):
            dpg.set_value("treatment_combo", row["treatment"])
        if pd.notnull(row["stain"]):
            dpg.set_value("stain_combo", row["stain"])
        if pd.notnull(row["treatment"]):
            dpg.set_value("treatment_combo", row["treatment"])

def z_slider_callback(sender, app_data, user_data):
    global gray_img
    if channel_zstack is None:  
        return
    z0, z1 = dpg.get_value("z_min_slider"), dpg.get_value("z_max_slider")
    gray_img = max_proj(channel_zstack[z0:z1+1])
    update_texture(gray_img, force=True)

def rip_checkbox_callback(sender, app_data, user_data):
    dpg.set_value("show_masks_checkbox", False)
    global mask_array, colors, selected_masks, texture_cache
    dpg.configure_item("run_rip_button", show=app_data)
    dpg.configure_item("show_masks_checkbox", show=app_data)
    if not app_data:
        mask_array = None
        colors.clear()
        selected_masks.clear()
        texture_cache = None
        update_texture(gray_img, force=True)

def save_metadata_callback(sender, app_data, user_data):
    global metadata_df
    if opened_file is None:
        dpg.set_value("status_text", "No file loaded.")
        return

    eye = dpg.get_value("eye_combo")
    time_str = dpg.get_value("time_input")
    djid = dpg.get_value("djid_input")
    treatment = dpg.get_value("treatment_combo")
    stain = dpg.get_value("stain_combo")

    if not djid.strip():
        dpg.set_value("status_text", "Please enter DJID.")
        return
    if not eye:
        dpg.set_value("status_text", "Please select an eye.")
        return
    if not stain:
        dpg.set_value("status_text", "Please select a stain.")
        return
    if not treatment:
        dpg.set_value("status_text", "Please select a treatment.")
        return
    if not time_str.strip():
        dpg.set_value("status_text", "Time condition is required.")
        return

    try:
        time_min = int(time_str)
    except ValueError:
        dpg.set_value("status_text", "Time must be an integer.")
        return

    z0 = dpg.get_value("z_min_slider")
    z1 = dpg.get_value("z_max_slider")

    data = {
        "filename": opened_file,
        "z_min": z0,
        "z_max": z1,
        "rip_cells": [],
        "eye": eye,
        "time_min": time_min,
        "djid": djid,
        "treatment": treatment,
        "stain": stain
    }

    if opened_file in metadata_df["filename"].values:
        idx = metadata_df["filename"] == opened_file
        metadata_df.loc[idx, :] = pd.DataFrame([data])
    else:
        metadata_df.loc[len(metadata_df)] = data

    dpg.set_value("status_text", f"Saved metadata for {opened_file}")
    refresh_contents_list()

    for display_name, real_name in display_map.items():
        if real_name == opened_file:
            dpg.set_value("contents_list", display_name)
            break

    dpg.show_item("rip_group")

def run_rip_detector_callback(sender, app_data, user_data):
    global metadata_df, mask_array, colors, selected_masks, texture_cache, gray_img

    if opened_file not in metadata_df["filename"].values:
        dpg.set_value("status_text", "Set Z boundaries before running rip detector.")
        return

    z0 = dpg.get_value("z_min_slider")
    z1 = dpg.get_value("z_max_slider")
    metadata_df.loc[metadata_df["filename"] == opened_file, ["z_min", "z_max"]] = [z0, z1]

    mask_array = None
    selected_masks.clear()
    colors.clear()
    texture_cache = None

    dpg.set_value("status_text", "Rip detection started...")
    dpg.show_item("show_masks_checkbox")
    dpg.set_value("show_masks_checkbox", True)
    dpg.hide_item("confirm_masks_button")
    dpg.hide_item("selected_mask_count")
    update_texture(gray_img, force=True)

    mp_DAPI = auto_brightness_contrast(gray_img)
    model_path = os.path.join(ROOT_DIR, 'CP_models', 'T5_DAPI_V4')
    deblur_model = denoise.CellposeDenoiseModel(gpu=True, model_type=model_path, restore_type="deblur_cyto3")
    masks, _, _, _ = deblur_model.eval(mp_DAPI, diameter=None, channels=[0, 0])

    if masks.max() == 0:
        dpg.set_value("status_text", "No masks detected.")
        return

    mask_array = masks - masks.min()
    colors.update({m: np.random.rand(3) for m in np.unique(mask_array) if m > 0})
    selected_masks.clear()
    update_texture(gray_img, force=True)

    dpg.set_value("selected_mask_count", "Cells in rip: []")
    dpg.show_item("selected_mask_count")
    dpg.show_item("confirm_masks_button")
    dpg.set_value("status_text", "Rip detection complete")

def wga_view_callback(sender, app_data, user_data):
    img = gray_img if not dpg.get_value("wga_checkbox") else channel2_stack[dpg.get_value("wga_slider")]
    update_texture(img, force=True)

def update_texture(base_img=None, force=False):
    global gray_img, channel2_stack, texture_cache, last_show_masks, last_selected, mask_array, selected_masks, colors
    print("\n[DEBUG] update_texture called")
    if base_img is None:
        base_img = gray_img if not dpg.get_value("wga_checkbox") else channel2_stack[dpg.get_value("wga_slider")]

    show_masks = dpg.get_value("show_masks_checkbox")
    if not force and texture_cache is not None and show_masks == last_show_masks and selected_masks == last_selected:
        _set_dynamic_texture_from_array(np.asarray(texture_cache).reshape((base_img.shape[0], base_img.shape[1], 4)))
        return

    print(f"[DEBUG] Image min: {np.min(base_img)}, max: {np.max(base_img)}")
    base = to_8bit(base_img).astype(np.float32) / 255.0
    rgba = np.zeros((*base_img.shape, 4), dtype=np.float32)
    rgba[..., :3] = base[..., None]
    rgba[..., 3] = 1.0

    if show_masks and mask_array is not None:
        print("\n[DEBUG] embedding mask outlines in update_texture")
        outline_rgba = np.zeros((*base_img.shape, 4), dtype=np.float32)
        for m in np.unique(mask_array):
            if m == 0:
                continue
            mask = (mask_array == m).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            col = colors.get(m, np.random.rand(3))
            for contour in contours:
                cv2.polylines(outline_rgba, [contour], isClosed=True, color=(*col, 1.0), thickness=1)
                if m in selected_masks:
                    cv2.fillPoly(outline_rgba, [contour], color=(*col, 0.4))
        mask_alpha = outline_rgba[..., 3:4]
        rgba[..., :3] = (1 - mask_alpha) * rgba[..., :3] + mask_alpha * outline_rgba[..., :3]
        rgba[..., 3] = np.clip(rgba[..., 3] + outline_rgba[..., 3], 0, 1)

    texture_cache = rgba.flatten().tolist()
    last_show_masks = show_masks
    last_selected = selected_masks.copy()
    _set_dynamic_texture_from_array(rgba)
    if selected_masks:
        dpg.set_value("selected_mask_count", f"Cells in rip: {sorted(selected_masks)}")
    else:
        dpg.set_value("selected_mask_count", "Cells in rip: []")

def mask_click_callback(sender, app_data, user_data):
    if not dpg.get_value("show_masks_checkbox"):
        return
    if mask_array is None or mask_array.max() == 0:
        return
    mx, my = dpg.get_mouse_pos(local=False)
    x0, y0 = dpg.get_item_rect_min("drawlist")
    ix, iy = int(mx - x0), int(my - y0)
    if ix < 0 or iy < 0 or ix >= gray_img.shape[1] or iy >= gray_img.shape[0]:
        return
    m = int(mask_array[iy, ix])
    if m > 0:
        if m in selected_masks:
            selected_masks.remove(m)
        else:
            selected_masks.append(m)
        update_texture(gray_img, force=True)
        dpg.set_value("status_text", f"Selected mask: {m}")
        dpg.set_value("selected_mask_count", f"Cells in rip: {sorted(selected_masks)}")

def add_z_range_widget(parent, depth):
    for tag in ["z_range_group", "z_min_slider", "z_max_slider", "set_boundaries_button", "save_metadata_button"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    mid = depth // 2
    with dpg.group(parent=parent, horizontal=False, tag="z_range_group"):
        dpg.add_text('Set Z-Boundaries')
        with dpg.group(horizontal=True):
            dpg.add_text("Min Z:")
            dpg.add_slider_int(label="", tag="z_min_slider", min_value=0, max_value=mid, default_value=0, callback=z_slider_callback, width = 264)
        with dpg.group(horizontal=True):
            dpg.add_text("Max Z:")
            dpg.add_slider_int(label="", tag="z_max_slider", min_value=mid, max_value=depth - 1, default_value=depth - 1, callback=z_slider_callback, width = 264)
        dpg.add_spacer(height=20)
        dpg.add_button(label="Save Info", tag="save_metadata_button", show=True, callback=save_metadata_callback, width=315)

def confirm_mask_selection_callback(sender, app_data, user_data):
    global metadata_df, selected_masks
    if opened_file in metadata_df["filename"].values:
        idx = metadata_df["filename"] == opened_file
        metadata_df.at[metadata_df.index[idx][0], "rip_cells"] = selected_masks.copy()
        dpg.set_value("selected_mask_count", f"Cells in rip: {sorted(selected_masks)}")
        dpg.set_value("status_text", f"Masks confirmed for {opened_file}")