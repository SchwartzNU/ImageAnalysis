import os
import json
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
CANVAS_WIDTH = 1024
CANVAS_HEIGHT = 512

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
segmentation_filtered_idxs_original = None
segmentation_npz_path = None
segmentation_display_rect = None
main_display_rect = None
segmentation_render_cache = None
segmentation_render_cache_settings = None
segmentation_base_rgba_cache = None
manual_excluded_masks = set()
METADATA_COLUMNS = ["filename", "z_min", "z_max", "rip_cells", "eye", "time_min", "djid", "treatment", "stain"]
metadata_df = pd.DataFrame(columns=METADATA_COLUMNS)
texture_cache = None
last_show_masks = True
last_selected = []
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
    # For now, only handle main image texture
    # Segmentation is handled directly in display_segmentation_filtered
    if texture_tag != "dynamic_texture":
        print(f"[WARN] _set_dynamic_texture_from_array called with unexpected texture_tag: {texture_tag}")
        return

    # Main image: scale to fit 1024x512 maintaining aspect ratio
    # Calculate scale to fit within 1024x512
    scale = min(CANVAS_WIDTH / w, CANVAS_HEIGHT / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad to 1024x512
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32)
    y_offset = (CANVAS_HEIGHT - new_h) // 2
    x_offset = (CANVAS_WIDTH - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    global main_display_rect
    main_display_rect = (x_offset, y_offset, new_w, new_h)
    
    # Flatten and update texture
    flat = canvas.flatten().tolist()
    try:
        dpg.set_value(texture_tag, flat)
    except Exception as exc:
        print(f"[WARN] set_value failed: {exc}")

def _fit_rgba_to_canvas(rgba, canvas_w=CANVAS_WIDTH, canvas_h=CANVAS_HEIGHT):
    """
    Scale RGBA content to fit inside a fixed canvas while preserving aspect ratio.
    Returns the padded canvas and the drawn rectangle as (x_offset, y_offset, new_w, new_h).
    """
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("rgba must be HxWx4")

    h, w = arr.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
    y_offset = (canvas_h - new_h) // 2
    x_offset = (canvas_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas, (x_offset, y_offset, new_w, new_h)

def _empty_metadata_df():
    return pd.DataFrame(columns=METADATA_COLUMNS)

def _metadata_file_path(folder=None):
    folder = folder or current_folder
    if not folder:
        return None
    folder_name = os.path.basename(folder.rstrip("/\\"))
    return os.path.join(folder, f"{folder_name}_gui_metadata.json")

def _normalize_rip_cells(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return []

def _normalize_metadata_record(record):
    return {
        "filename": str(record.get("filename", "")).strip(),
        "z_min": int(record.get("z_min", 0) or 0),
        "z_max": int(record.get("z_max", 0) or 0),
        "rip_cells": _normalize_rip_cells(record.get("rip_cells", [])),
        "eye": str(record.get("eye", "") or ""),
        "time_min": int(record.get("time_min", 0) or 0),
        "djid": str(record.get("djid", "") or ""),
        "treatment": str(record.get("treatment", "") or ""),
        "stain": str(record.get("stain", "") or ""),
    }

def _load_metadata_for_folder(folder=None):
    global metadata_df
    path = _metadata_file_path(folder)
    if not path or not os.path.exists(path):
        metadata_df = _empty_metadata_df()
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = []
        if isinstance(payload, list):
            for record in payload:
                if isinstance(record, dict):
                    normalized = _normalize_metadata_record(record)
                    if normalized["filename"]:
                        rows.append(normalized)
        metadata_df = pd.DataFrame(rows, columns=METADATA_COLUMNS) if rows else _empty_metadata_df()
        print(f"[DEBUG] Loaded {len(metadata_df)} metadata rows from {path}")
    except Exception as exc:
        metadata_df = _empty_metadata_df()
        print(f"[WARN] Failed to load metadata file {path}: {exc}")

def _persist_metadata():
    path = _metadata_file_path()
    if not path:
        return

    rows = []
    for record in metadata_df.to_dict(orient="records"):
        normalized = _normalize_metadata_record(record)
        if normalized["filename"]:
            rows.append(normalized)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[DEBUG] Saved {len(rows)} metadata rows to {path}")

def _restore_contents_selection():
    if opened_file is None:
        return
    for display_name, real_name in display_map.items():
        if real_name == opened_file:
            dpg.set_value("contents_list", display_name)
            break

def _clear_segmentation_render_cache():
    global segmentation_render_cache, segmentation_render_cache_settings, segmentation_base_rgba_cache
    segmentation_render_cache = None
    segmentation_render_cache_settings = None
    segmentation_base_rgba_cache = None

def _get_cached_segmentation_render(show_removed, label_filtered_only):
    global segmentation_render_cache, segmentation_render_cache_settings
    if segmentation_masks is None:
        return None

    filtered_key = None if segmentation_filtered_idxs is None else tuple(int(v) for v in segmentation_filtered_idxs)
    settings = (bool(show_removed), bool(label_filtered_only), filtered_key)
    if segmentation_render_cache is not None and segmentation_render_cache_settings == settings:
        return segmentation_render_cache

    import analysis_helpers
    segmentation_render_cache = analysis_helpers.create_labeled_segmentation_image(
        segmentation_masks,
        segmentation_colors,
        filtered_idxs=segmentation_filtered_idxs,
        label_min_area=0,
        force_labels=True,
        show_removed=show_removed,
        label_filtered_only=label_filtered_only,
    )
    segmentation_render_cache_settings = settings
    return segmentation_render_cache

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
    _load_metadata_for_folder(folder)
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
    _load_metadata_for_folder(folder)
    two_level = f"{os.path.basename(os.path.dirname(folder))}\\{os.path.basename(folder)}"
    dpg.set_value("dir_path_repeat", two_level)
    refresh_contents_list()
    for tag in ("z_range_group","rip_group","wga_group"):
        if dpg.does_item_exist(tag):
            dpg.hide_item(tag)
    dpg.set_value("status_text", "Folder loaded")

def _delete_segmentation_outputs_for_file(filename):
    if current_folder is None or not filename:
        return
    folder_name = os.path.basename(current_folder.rstrip("/\\"))
    segmentation_dir = os.path.join(current_folder, f"{folder_name}_segmentation")
    if not os.path.exists(segmentation_dir):
        return
    base_name = os.path.splitext(filename)[0]
    for f in os.listdir(segmentation_dir):
        if f.startswith(base_name) and f.endswith((".npz", ".png", ".tif", ".tiff")):
            try:
                os.remove(os.path.join(segmentation_dir, f))
            except Exception:
                pass

def load_segmentation_if_available(filename):
    """
    Load segmentation masks and assign colors if available.
    """
    global segmentation_masks, segmentation_colors, segmentation_filtered_idxs
    global segmentation_filtered_idxs_original, segmentation_npz_path, manual_excluded_masks
    _clear_segmentation_render_cache()
    
    # Clear any previous segmentation display immediately
    if dpg.does_item_exist("segmentation_window"):
        try:
            dpg.hide_item("segmentation_window")
        except Exception:
            pass
    # Clear segmentation texture to blank
    try:
        if dpg.does_item_exist("segmentation_texture"):
            blank = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32).flatten().tolist()
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
    
    if os.path.exists(seg_file):
        try:
            import analysis_helpers
            seg_data = np.load(seg_file, allow_pickle=True)
            segmentation_masks = seg_data['dapi_masks']
            segmentation_filtered_idxs = list(seg_data['filtered_idxs'])
            segmentation_filtered_idxs_original = list(segmentation_filtered_idxs)
            segmentation_npz_path = seg_file
            manual_excluded_masks.clear()
            if 'color_assignment' in seg_data.files:
                try:
                    segmentation_colors = seg_data['color_assignment'].item()
                    print(f"[DEBUG] Loaded {len(segmentation_colors)} colors from npz")
                except Exception:
                    segmentation_colors = analysis_helpers.assign_colors_to_masks(segmentation_masks)
                    print(f"[DEBUG] Recomputed {len(segmentation_colors)} colors after color_assignment load failed")
            else:
                segmentation_colors = analysis_helpers.assign_colors_to_masks(segmentation_masks)
                print(f"[DEBUG] Recomputed {len(segmentation_colors)} colors for visualization")
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
        segmentation_filtered_idxs_original = None
        segmentation_npz_path = None
        manual_excluded_masks.clear()

def display_segmentation_filtered():
    """
    Display the loaded segmentation with colors and labels, showing only filtered masks.
    """
    global segmentation_masks, segmentation_colors, segmentation_filtered_idxs
    global segmentation_display_rect, segmentation_base_rgba_cache
    
    if segmentation_masks is None:
        if dpg.does_item_exist("segmentation_window"):
            dpg.hide_item("segmentation_window")
        return
    
    # Read user preference for showing removed masks
    show_removed = dpg.get_value("show_removed_masks") if dpg.does_item_exist("show_removed_masks") else False
    label_filtered_only = dpg.get_value("label_filtered_only") if dpg.does_item_exist("label_filtered_only") else True
    overlay_on_image = dpg.get_value("seg_overlay_on_image") if dpg.does_item_exist("seg_overlay_on_image") else False
    overlay_alpha = dpg.get_value("seg_overlay_alpha") if dpg.does_item_exist("seg_overlay_alpha") else 0.45

    seg_viz = _get_cached_segmentation_render(show_removed, label_filtered_only)
    if seg_viz is None:
        return

    if overlay_on_image and gray_img is not None:
        if segmentation_base_rgba_cache is None:
            base = to_8bit(gray_img).astype(np.float32) / 255.0
            segmentation_base_rgba_cache = np.zeros((base.shape[0], base.shape[1], 4), dtype=np.float32)
            segmentation_base_rgba_cache[..., :3] = base[..., None]
            segmentation_base_rgba_cache[..., 3] = 1.0
        seg_alpha = np.clip(seg_viz[..., 3:4], 0.0, 1.0) * float(np.clip(overlay_alpha, 0.0, 1.0))
        blended = segmentation_base_rgba_cache.copy()
        blended[..., :3] = (1.0 - seg_alpha) * blended[..., :3] + seg_alpha * seg_viz[..., :3]
        blended[..., 3] = 1.0
        canvas, segmentation_display_rect = _fit_rgba_to_canvas(blended)
    else:
        canvas, segmentation_display_rect = _fit_rgba_to_canvas(seg_viz)
    
    # Update texture directly
    flat = canvas.flatten().tolist()
    try:
        if dpg.does_item_exist("segmentation_texture"):
            dpg.set_value("segmentation_texture", flat)
        if dpg.does_item_exist("segmentation_window"):
            dpg.show_item("segmentation_window")
    except Exception as e:
        print(f"[ERROR] updating segmentation texture: {e}")
        import traceback
        traceback.print_exc()

def _recompute_filtered_from_manual():
    if segmentation_masks is None:
        return None
    ids = np.unique(segmentation_masks)
    ids = ids[ids > 0]
    keep = [int(mid) - 1 for mid in ids if int(mid) not in manual_excluded_masks]
    return keep

def segmentation_click_callback(sender, app_data, user_data):
    if not dpg.does_item_exist("manual_filter_mode") or not dpg.get_value("manual_filter_mode"):
        return
    if segmentation_masks is None or segmentation_display_rect is None:
        return
    mx, my = dpg.get_mouse_pos(local=False)
    x0, y0 = dpg.get_item_rect_min("segmentation_drawlist")
    ix, iy = int(mx - x0), int(my - y0)
    draw_w, draw_h = dpg.get_item_rect_size("segmentation_drawlist")
    if draw_w <= 0 or draw_h <= 0 or ix < 0 or iy < 0 or ix >= draw_w or iy >= draw_h:
        return
    tex_x = ix * (CANVAS_WIDTH / float(draw_w))
    tex_y = iy * (CANVAS_HEIGHT / float(draw_h))
    disp_x, disp_y, disp_w, disp_h = segmentation_display_rect
    if tex_x < disp_x or tex_y < disp_y or tex_x >= disp_x + disp_w or tex_y >= disp_y + disp_h:
        return
    h, w = segmentation_masks.shape
    x = int((tex_x - disp_x) * (w / float(disp_w)))
    y = int((tex_y - disp_y) * (h / float(disp_h)))
    if x < 0 or y < 0 or x >= w or y >= h:
        return
    mask_id = int(segmentation_masks[y, x])
    if mask_id <= 0:
        return
    if mask_id in manual_excluded_masks:
        manual_excluded_masks.remove(mask_id)
    else:
        manual_excluded_masks.add(mask_id)
    new_filtered = _recompute_filtered_from_manual()
    if new_filtered is not None:
        global segmentation_filtered_idxs
        segmentation_filtered_idxs = new_filtered
    display_segmentation_filtered()

def apply_manual_filter(sender=None, app_data=None, user_data=None):
    global segmentation_filtered_idxs_original
    if segmentation_npz_path is None or segmentation_masks is None:
        return
    new_filtered = _recompute_filtered_from_manual()
    if new_filtered is None:
        return
    try:
        seg_data = np.load(segmentation_npz_path, allow_pickle=True)
        payload = {k: seg_data[k] for k in seg_data.files}
        payload["filtered_idxs"] = np.array(new_filtered, dtype=int)
        np.savez(segmentation_npz_path, **payload)
        segmentation_filtered_idxs_original = list(new_filtered)
        _clear_segmentation_render_cache()
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", "Manual filter saved to segmentation file")
    except Exception as exc:
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", f"Failed to save manual filter: {exc}")

def reset_manual_filter(sender=None, app_data=None, user_data=None):
    manual_excluded_masks.clear()
    if segmentation_filtered_idxs_original is not None:
        global segmentation_filtered_idxs
        segmentation_filtered_idxs = list(segmentation_filtered_idxs_original)
    _clear_segmentation_render_cache()
    display_segmentation_filtered()

def display_segmentation():
    """
    Display the loaded segmentation with colors and labels.
    """
    global gray_img, segmentation_masks, segmentation_colors, segmentation_display_rect
    
    if segmentation_masks is None:
        return
    
    seg_viz = _get_cached_segmentation_render(show_removed=False, label_filtered_only=False)
    if seg_viz is None:
        return
    canvas, segmentation_display_rect = _fit_rgba_to_canvas(seg_viz)
    
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
        _clear_segmentation_render_cache()
        
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

        # Load any saved segmentation for this file by default
        load_segmentation_if_available(sel)

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
    global gray_img, segmentation_base_rgba_cache
    if channel_zstack is None:  
        return
    z0, z1 = dpg.get_value("z_min_slider"), dpg.get_value("z_max_slider")
    gray_img = max_proj(channel_zstack[z0:z1+1])
    segmentation_base_rgba_cache = None
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
    existing_rip_cells = []
    if opened_file in metadata_df["filename"].values:
        existing_rip_cells = metadata_df.loc[metadata_df["filename"] == opened_file, "rip_cells"].iloc[0]

    data = {
        "filename": opened_file,
        "z_min": z0,
        "z_max": z1,
        "rip_cells": _normalize_rip_cells(existing_rip_cells),
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

    _persist_metadata()
    dpg.set_value("status_text", f"Saved metadata for {opened_file}")
    refresh_contents_list()
    _restore_contents_selection()

    dpg.show_item("rip_group")

def run_rip_detector_callback(sender, app_data, user_data):
    global metadata_df, mask_array, colors, selected_masks, texture_cache, gray_img

    if opened_file not in metadata_df["filename"].values:
        dpg.set_value("status_text", "Set Z boundaries before running rip detector.")
        return

    z0 = dpg.get_value("z_min_slider")
    z1 = dpg.get_value("z_max_slider")
    metadata_df.loc[metadata_df["filename"] == opened_file, ["z_min", "z_max"]] = [z0, z1]
    _persist_metadata()

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
    if mask_array is None or mask_array.max() == 0 or gray_img is None or main_display_rect is None:
        return
    mx, my = dpg.get_mouse_pos(local=False)
    x0, y0 = dpg.get_item_rect_min("drawlist")
    ix, iy = int(mx - x0), int(my - y0)
    draw_w, draw_h = dpg.get_item_rect_size("drawlist")
    if draw_w <= 0 or draw_h <= 0 or ix < 0 or iy < 0 or ix >= draw_w or iy >= draw_h:
        return
    tex_x = ix * (CANVAS_WIDTH / float(draw_w))
    tex_y = iy * (CANVAS_HEIGHT / float(draw_h))
    disp_x, disp_y, disp_w, disp_h = main_display_rect
    if tex_x < disp_x or tex_y < disp_y or tex_x >= disp_x + disp_w or tex_y >= disp_y + disp_h:
        return
    img_x = int((tex_x - disp_x) * (gray_img.shape[1] / float(disp_w)))
    img_y = int((tex_y - disp_y) * (gray_img.shape[0] / float(disp_h)))
    if img_x < 0 or img_y < 0 or img_x >= gray_img.shape[1] or img_y >= gray_img.shape[0]:
        return
    m = int(mask_array[img_y, img_x])
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
        _persist_metadata()
        refresh_contents_list()
        _restore_contents_selection()
        dpg.set_value("selected_mask_count", f"Cells in rip: {sorted(selected_masks)}")
        dpg.set_value("status_text", f"Masks confirmed for {opened_file}")
