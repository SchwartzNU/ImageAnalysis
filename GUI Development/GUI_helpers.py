import os
import json
import re
import warnings
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
GFP_REVIEW_CACHE_VERSION = 2

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

current_folder = None
opened_file = None
channel_zstack = None
channel2_stack = None
channel_gfp_stack = None
opened_file_channel_indices = None
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
gfp_review_projection = None
gfp_review_overlay_masks = None
gfp_review_intensity_by_mask = {}
gfp_review_render_regions = []
gfp_review_segmentation_regions = []
gfp_review_display_rect = None
gfp_review_cache_path = None
gfp_review_filename = None
gfp_review_source_label = ""
gfp_review_prepare_hint = ""
METADATA_COLUMNS = [
    "filename",
    "z_min",
    "z_max",
    "rip_cells",
    "eye",
    "time_min",
    "djid",
    "treatment",
    "stain",
    "egfp_threshold",
    "egfp_reviewed",
]
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

def has_gfp_channel_enabled():
    return dpg.get_value("has_gfp_channel") if dpg.does_item_exist("has_gfp_channel") else True

def get_image_channel_indices(num_channels):
    use_gfp = has_gfp_channel_enabled() and num_channels >= 4
    if use_gfp:
        return {"dapi": 0, "egfp": 1, "wga": 2, "stain": 3}
    if num_channels >= 3:
        return {"dapi": 0, "egfp": None, "wga": 1, "stain": 2}
    raise ValueError(f"Expected at least 3 channels, found {num_channels}")

def gfp_channel_toggle_callback(sender=None, app_data=None, user_data=None):
    global opened_file, opened_file_channel_indices
    if opened_file is None or not dpg.does_item_exist("contents_list"):
        return

    current_file = opened_file
    display_name = _display_name_for_file(current_file)
    if display_name:
        dpg.set_value("contents_list", display_name)
    opened_file = None
    opened_file_channel_indices = None
    open_nd2_callback(sender, app_data, user_data)

def _display_name_for_file(filename):
    if not filename:
        return None
    for display_name, real_name in display_map.items():
        if real_name == filename:
            return display_name
    return filename

def _selected_filename_from_ui(app_data=None):
    display_name = app_data if isinstance(app_data, str) else None
    if not display_name and dpg.does_item_exist("contents_list"):
        value = dpg.get_value("contents_list")
        if isinstance(value, str):
            display_name = value
    if not display_name:
        return None
    return display_map.get(display_name, display_name)

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
    egfp_threshold = record.get("egfp_threshold", np.nan)
    try:
        egfp_threshold = float(egfp_threshold)
    except (TypeError, ValueError):
        egfp_threshold = np.nan

    egfp_reviewed = record.get("egfp_reviewed", False)
    if isinstance(egfp_reviewed, str):
        egfp_reviewed = egfp_reviewed.strip().lower() in {"1", "true", "yes"}

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
        "egfp_threshold": egfp_threshold,
        "egfp_reviewed": bool(egfp_reviewed),
    }

def _list_nd2_files(folder=None):
    folder = folder or current_folder
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if isinstance(f, str) and f.lower().endswith(".nd2")
    )

def _metadata_filename_key(filename):
    stem = os.path.splitext(os.path.basename(str(filename or "")))[0].strip().lower()
    return re.sub(r"_v\d+$", "", stem)

def reconcile_metadata_with_folder(folder=None, persist=False):
    global metadata_df
    folder = folder or current_folder
    actual_files = _list_nd2_files(folder)
    actual_lookup = {name.lower(): name for name in actual_files}
    file_order = {name.lower(): idx for idx, name in enumerate(actual_files)}

    rows = []
    for record in metadata_df.to_dict(orient="records"):
        normalized = _normalize_metadata_record(record)
        if normalized["filename"]:
            rows.append(normalized)

    reconciled = []
    used_actual = set()
    unmatched_rows = []
    changed = False

    for record in rows:
        actual_name = actual_lookup.get(record["filename"].lower())
        if actual_name is None:
            unmatched_rows.append(record)
            changed = True
            continue
        actual_key = actual_name.lower()
        if actual_key in used_actual:
            changed = True
            continue
        normalized = dict(record)
        if normalized["filename"] != actual_name:
            normalized["filename"] = actual_name
            changed = True
        reconciled.append(normalized)
        used_actual.add(actual_key)

    for actual_name in actual_files:
        actual_key = actual_name.lower()
        if actual_key in used_actual:
            continue
        match_idx = next(
            (
                idx for idx, record in enumerate(unmatched_rows)
                if _metadata_filename_key(record["filename"]) == _metadata_filename_key(actual_name)
            ),
            None,
        )
        if match_idx is None:
            continue
        normalized = dict(unmatched_rows.pop(match_idx))
        if normalized["filename"] != actual_name:
            normalized["filename"] = actual_name
            changed = True
        reconciled.append(_normalize_metadata_record(normalized))
        used_actual.add(actual_key)

    reconciled.sort(key=lambda record: file_order.get(record["filename"].lower(), len(file_order)))
    metadata_df = pd.DataFrame(reconciled, columns=METADATA_COLUMNS) if reconciled else _empty_metadata_df()

    if persist and folder:
        _persist_metadata()

    return changed

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
        reconcile_metadata_with_folder(folder, persist=bool(rows))
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

def _restore_contents_selection():
    if opened_file is None or not dpg.does_item_exist("contents_list"):
        return
    display_name = _display_name_for_file(opened_file)
    if display_name and dpg.get_value("contents_list") != display_name:
        dpg.set_value("contents_list", display_name)

def _clear_segmentation_render_cache():
    global segmentation_render_cache, segmentation_render_cache_settings, segmentation_base_rgba_cache
    segmentation_render_cache = None
    segmentation_render_cache_settings = None
    segmentation_base_rgba_cache = None

def _clear_gfp_review_state():
    global gfp_review_projection, gfp_review_overlay_masks, gfp_review_intensity_by_mask
    global gfp_review_render_regions, gfp_review_segmentation_regions
    global gfp_review_display_rect, gfp_review_cache_path, gfp_review_filename
    global gfp_review_source_label, gfp_review_prepare_hint
    gfp_review_projection = None
    gfp_review_overlay_masks = None
    gfp_review_intensity_by_mask = {}
    gfp_review_render_regions = []
    gfp_review_segmentation_regions = []
    gfp_review_display_rect = None
    gfp_review_cache_path = None
    gfp_review_filename = None
    gfp_review_source_label = ""
    gfp_review_prepare_hint = ""

def _segmentation_file_path(filename):
    if current_folder is None or not filename:
        return None
    folder_name = os.path.basename(current_folder.rstrip("/\\"))
    segmentation_dir = os.path.join(current_folder, f"{folder_name}_segmentation")
    base_name = os.path.splitext(filename)[0]
    return os.path.join(segmentation_dir, f"{base_name}_segmentation.npz")

def _gfp_review_file_path(filename):
    if current_folder is None or not filename:
        return None
    folder_name = os.path.basename(current_folder.rstrip("/\\"))
    segmentation_dir = os.path.join(current_folder, f"{folder_name}_segmentation")
    base_name = os.path.splitext(filename)[0]
    return os.path.join(segmentation_dir, f"{base_name}_gfp_review.npz")

def invalidate_gfp_review_state(filename, remove_cache=False, clear_loaded=False, persist=True):
    global metadata_df
    if not filename:
        return False

    changed = False
    if filename in metadata_df["filename"].values:
        idx = metadata_df["filename"] == filename
        current_threshold = pd.to_numeric(metadata_df.loc[idx, "egfp_threshold"], errors="coerce")
        current_reviewed = metadata_df.loc[idx, "egfp_reviewed"]
        reviewed_true = current_reviewed.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
        if current_threshold.notna().any() or reviewed_true.any():
            metadata_df.loc[idx, "egfp_threshold"] = np.nan
            metadata_df.loc[idx, "egfp_reviewed"] = False
            changed = True

    if remove_cache:
        review_path = _gfp_review_file_path(filename)
        if review_path and os.path.exists(review_path):
            try:
                os.remove(review_path)
            except OSError:
                pass

    if clear_loaded and filename == gfp_review_filename:
        _clear_gfp_review_state()
        if dpg.does_item_exist("gfp_review_window"):
            dpg.hide_item("gfp_review_window")

    if changed and persist:
        _persist_metadata()
    refresh_gfp_review_controls()
    return changed

def gfp_channel_exists_for_opened_file():
    global opened_file_channel_indices
    if opened_file is None or current_folder is None:
        return False
    if isinstance(opened_file_channel_indices, dict):
        return opened_file_channel_indices.get("egfp") is not None
    try:
        path = os.path.join(current_folder, opened_file)
        with nd2.ND2File(path) as f:
            opened_file_channel_indices = get_image_channel_indices(int(f.sizes.get("C", 0)))
        return opened_file_channel_indices.get("egfp") is not None
    except Exception:
        opened_file_channel_indices = None
        return False

def _configure_gfp_threshold_slider(min_val, max_val, value):
    if not dpg.does_item_exist("gfp_threshold_slider"):
        return
    if not np.isfinite(min_val):
        min_val = 0.0
    if not np.isfinite(max_val):
        max_val = max(min_val + 1.0, 1.0)
    if max_val <= min_val:
        max_val = min_val + 1.0
    value = float(np.clip(value, min_val, max_val))
    dpg.configure_item("gfp_threshold_slider", min_value=float(min_val), max_value=float(max_val))
    dpg.set_value("gfp_threshold_slider", value)
    if dpg.does_item_exist("gfp_threshold_value"):
        dpg.set_value("gfp_threshold_value", f"Threshold: {value:.2f}")

def refresh_gfp_review_controls():
    file_loaded = opened_file is not None and current_folder is not None
    has_gfp = file_loaded and gfp_channel_exists_for_opened_file()
    segmentation_path = _segmentation_file_path(opened_file) if file_loaded else None
    has_segmentation = bool(segmentation_path and os.path.exists(segmentation_path))
    review_loaded = gfp_review_projection is not None and gfp_review_overlay_masks is not None

    if dpg.does_item_exist("prepare_gfp_review_button"):
        dpg.configure_item("prepare_gfp_review_button", enabled=bool(has_gfp and has_segmentation))
    if dpg.does_item_exist("accept_gfp_review_button"):
        dpg.configure_item("accept_gfp_review_button", enabled=bool(review_loaded))
    if dpg.does_item_exist("gfp_threshold_slider"):
        dpg.configure_item("gfp_threshold_slider", enabled=bool(review_loaded))

def _update_gfp_status():
    global gfp_review_source_label, gfp_review_prepare_hint
    if not dpg.does_item_exist("gfp_status_text"):
        return
    if opened_file is None:
        dpg.set_value("gfp_status_text", "GFP Status: Load a file first.")
        return
    if not gfp_channel_exists_for_opened_file():
        dpg.set_value("gfp_status_text", "GFP Status: Current file does not include a GFP channel.")
        return
    if _segmentation_file_path(opened_file) is None or not os.path.exists(_segmentation_file_path(opened_file)):
        dpg.set_value("gfp_status_text", "GFP Status: Run segmentation first.")
        return
    if gfp_review_projection is None or gfp_review_overlay_masks is None:
        if gfp_review_prepare_hint:
            dpg.set_value("gfp_status_text", f"GFP Status: {gfp_review_prepare_hint}")
        else:
            dpg.set_value("gfp_status_text", "GFP Status: Prepare GFP review for this file.")
        return

    threshold = dpg.get_value("gfp_threshold_slider") if dpg.does_item_exist("gfp_threshold_slider") else 0.0
    positive = 0
    total = 0
    for mask_id, raw_val in gfp_review_intensity_by_mask.items():
        total += 1
        if raw_val >= threshold:
            positive += 1
    accepted_suffix = ""
    if opened_file in metadata_df["filename"].values:
        row = metadata_df.loc[metadata_df["filename"] == opened_file].iloc[0]
        reviewed = row.get("egfp_reviewed", False)
        if isinstance(reviewed, str):
            reviewed = reviewed.strip().lower() in {"1", "true", "yes"}
        if bool(reviewed):
            accepted_suffix = " (accepted)"
    source_suffix = f" [{gfp_review_source_label}]" if gfp_review_source_label else ""
    dpg.set_value("gfp_status_text", f"GFP Status: {positive} / {total} cells positive at threshold {float(threshold):.2f}{accepted_suffix}{source_suffix}")

def _rebuild_gfp_review_render_regions():
    global gfp_review_render_regions
    gfp_review_render_regions = []
    if gfp_review_overlay_masks is None:
        return

    labels = np.unique(gfp_review_overlay_masks)
    labels = labels[labels > 0]
    for label in labels:
        ys, xs = np.where(gfp_review_overlay_masks == label)
        if ys.size == 0 or xs.size == 0:
            continue
        row0, row1 = int(ys.min()), int(ys.max()) + 1
        col0, col1 = int(xs.min()), int(xs.max()) + 1
        local_mask = (gfp_review_overlay_masks[row0:row1, col0:col1] == label).astype(np.uint8)
        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gfp_review_render_regions.append({
            "label": int(label),
            "row0": row0,
            "row1": row1,
            "col0": col0,
            "col1": col1,
            "mask": local_mask.astype(bool),
            "contours": contours,
        })

def _rebuild_gfp_segmentation_regions():
    global gfp_review_segmentation_regions
    gfp_review_segmentation_regions = []
    if segmentation_masks is None:
        return

    filtered_set = set(int(idx) for idx in segmentation_filtered_idxs) if segmentation_filtered_idxs is not None else None
    labels = np.unique(segmentation_masks)
    labels = labels[labels > 0]
    for label in labels:
        if filtered_set is not None and (int(label) - 1) not in filtered_set:
            continue
        ys, xs = np.where(segmentation_masks == label)
        if ys.size == 0 or xs.size == 0:
            continue
        row0, row1 = int(ys.min()), int(ys.max()) + 1
        col0, col1 = int(xs.min()), int(xs.max()) + 1
        local_mask = (segmentation_masks[row0:row1, col0:col1] == label).astype(np.uint8)
        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gfp_review_segmentation_regions.append({
            "row0": row0,
            "row1": row1,
            "col0": col0,
            "col1": col1,
            "contours": contours,
        })

def display_gfp_review():
    global gfp_review_display_rect
    if gfp_review_projection is None or gfp_review_overlay_masks is None:
        if dpg.does_item_exist("gfp_review_window"):
            dpg.hide_item("gfp_review_window")
        return

    base = to_8bit(gfp_review_projection).astype(np.float32) / 255.0
    rgba = np.zeros((base.shape[0], base.shape[1], 4), dtype=np.float32)
    rgba[..., :3] = base[..., None]
    rgba[..., 3] = 1.0

    threshold = dpg.get_value("gfp_threshold_slider") if dpg.does_item_exist("gfp_threshold_slider") else 0.0
    for region in gfp_review_render_regions:
        label = region["label"]
        mask = region["mask"]
        raw_val = gfp_review_intensity_by_mask.get(label, np.nan)
        is_positive = np.isfinite(raw_val) and raw_val >= threshold
        fill_color = np.array([0.10, 0.85, 0.25], dtype=np.float32) if is_positive else np.array([0.85, 0.85, 0.85], dtype=np.float32)
        alpha = 0.30 if is_positive else 0.08
        row0, row1 = region["row0"], region["row1"]
        col0, col1 = region["col0"], region["col1"]
        patch = rgba[row0:row1, col0:col1]
        patch[mask, :3] = (1.0 - alpha) * patch[mask, :3] + alpha * fill_color

        outline_rgba = np.zeros_like(patch)
        outline_color = (0.05, 1.0, 0.2, 1.0) if is_positive else (0.8, 0.2, 0.2, 0.8)
        for contour in region["contours"]:
            cv2.polylines(outline_rgba, [contour], isClosed=True, color=outline_color, thickness=1)
        outline_alpha = outline_rgba[..., 3:4]
        patch[..., :3] = (1.0 - outline_alpha) * patch[..., :3] + outline_alpha * outline_rgba[..., :3]

    for region in gfp_review_segmentation_regions:
        row0, row1 = region["row0"], region["row1"]
        col0, col1 = region["col0"], region["col1"]
        patch = rgba[row0:row1, col0:col1]
        outline_rgba = np.zeros_like(patch)
        for contour in region["contours"]:
            cv2.polylines(outline_rgba, [contour], isClosed=True, color=(0.0, 1.0, 1.0, 1.0), thickness=2)
        outline_alpha = outline_rgba[..., 3:4]
        patch[..., :3] = (1.0 - outline_alpha) * patch[..., :3] + outline_alpha * outline_rgba[..., :3]

    canvas, gfp_review_display_rect = _fit_rgba_to_canvas(rgba)
    flat = canvas.flatten().tolist()
    if dpg.does_item_exist("gfp_review_texture"):
        dpg.set_value("gfp_review_texture", flat)
    if dpg.does_item_exist("gfp_review_window"):
        dpg.show_item("gfp_review_window")
    _update_gfp_status()

def gfp_threshold_slider_callback(sender=None, app_data=None, user_data=None):
    if dpg.does_item_exist("gfp_threshold_value"):
        dpg.set_value("gfp_threshold_value", f"Threshold: {float(app_data):.2f}")
    display_gfp_review()

def load_gfp_review_if_available(filename, source_label="cached"):
    global gfp_review_projection, gfp_review_overlay_masks, gfp_review_intensity_by_mask
    global gfp_review_cache_path, gfp_review_filename, gfp_review_source_label, gfp_review_prepare_hint
    _clear_gfp_review_state()

    if current_folder is None or not filename:
        refresh_gfp_review_controls()
        return
    review_path = _gfp_review_file_path(filename)
    if review_path is None or not os.path.exists(review_path):
        if dpg.does_item_exist("gfp_review_window"):
            dpg.hide_item("gfp_review_window")
        refresh_gfp_review_controls()
        _update_gfp_status()
        return

    review_data = np.load(review_path, allow_pickle=True)
    review_version = int(review_data["review_version"]) if "review_version" in review_data.files else 0
    if review_version != GFP_REVIEW_CACHE_VERSION:
        gfp_review_prepare_hint = f"Cached review is outdated. Press Prepare GFP Review to rebuild it."
        if dpg.does_item_exist("gfp_review_window"):
            dpg.hide_item("gfp_review_window")
        refresh_gfp_review_controls()
        _update_gfp_status()
        return
    gfp_review_projection = review_data["gfp_proj"]
    gfp_review_overlay_masks = review_data["overlay_masks"]
    mask_ids = review_data["mask_ids"].astype(int).tolist()
    raw_vals = review_data["egfp_raw_intensities"].astype(float).tolist()
    gfp_review_intensity_by_mask = {int(mask_id): float(raw_val) for mask_id, raw_val in zip(mask_ids, raw_vals)}
    _rebuild_gfp_review_render_regions()
    gfp_review_cache_path = review_path
    gfp_review_filename = filename
    gfp_review_source_label = source_label
    gfp_review_prepare_hint = ""

    default_threshold = float(review_data["default_threshold"]) if "default_threshold" in review_data.files else 0.0
    threshold_value = default_threshold
    if filename in metadata_df["filename"].values:
        row = metadata_df.loc[metadata_df["filename"] == filename].iloc[0]
        saved_threshold = pd.to_numeric(row.get("egfp_threshold", np.nan), errors="coerce")
        if np.isfinite(saved_threshold):
            threshold_value = float(saved_threshold)

    if raw_vals:
        slider_min = float(np.min(raw_vals))
        slider_max = float(np.max(raw_vals))
    else:
        slider_min, slider_max = 0.0, 1.0
    _configure_gfp_threshold_slider(slider_min, slider_max, threshold_value)
    refresh_gfp_review_controls()
    display_gfp_review()

def prepare_gfp_review_callback(sender=None, app_data=None, user_data=None):
    global gfp_review_prepare_hint
    if opened_file is None or current_folder is None:
        if dpg.does_item_exist("gfp_status_text"):
            dpg.set_value("gfp_status_text", "GFP Status: Load a file first.")
        return
    if not gfp_channel_exists_for_opened_file():
        if dpg.does_item_exist("gfp_status_text"):
            dpg.set_value("gfp_status_text", "GFP Status: Current file does not include a GFP channel.")
        return

    folder_name = os.path.basename(current_folder.rstrip("/\\"))
    segmentation_dir = os.path.join(current_folder, f"{folder_name}_segmentation")
    base_name = os.path.splitext(opened_file)[0]
    seg_file = os.path.join(segmentation_dir, f"{base_name}_segmentation.npz")
    if not os.path.exists(seg_file):
        if dpg.does_item_exist("gfp_status_text"):
            dpg.set_value("gfp_status_text", "GFP Status: Run segmentation first.")
        return

    if dpg.does_item_exist("prepare_gfp_review_button"):
        dpg.configure_item("prepare_gfp_review_button", enabled=False)
    if dpg.does_item_exist("gfp_status_text"):
        dpg.set_value("gfp_status_text", "GFP Status: Preparing GFP review...")
    gfp_review_prepare_hint = ""
    if dpg.is_dearpygui_running():
        dpg.split_frame(delay=1)

    try:
        import analysis_helpers
        file_path = os.path.join(current_folder, opened_file)
        review_path = _gfp_review_file_path(opened_file)
        invalidate_gfp_review_state(opened_file, remove_cache=False, clear_loaded=False, persist=True)
        analysis_helpers.prepare_gfp_review_data(
            file_path,
            seg_file,
            review_path,
            dapi_stack=channel_zstack,
            wga_stack=channel2_stack,
            egfp_stack=channel_gfp_stack,
        )
        load_gfp_review_if_available(opened_file, source_label="prepared now")
    except Exception as exc:
        if dpg.does_item_exist("gfp_status_text"):
            dpg.set_value("gfp_status_text", f"GFP Status: Error - {exc}")
    finally:
        if dpg.does_item_exist("prepare_gfp_review_button"):
            dpg.configure_item("prepare_gfp_review_button", enabled=True)
        refresh_gfp_review_controls()

def accept_gfp_review_callback(sender=None, app_data=None, user_data=None):
    global metadata_df
    if opened_file is None or gfp_review_projection is None:
        _update_gfp_status()
        return

    threshold = dpg.get_value("gfp_threshold_slider") if dpg.does_item_exist("gfp_threshold_slider") else np.nan
    if opened_file in metadata_df["filename"].values:
        idx = metadata_df["filename"] == opened_file
        metadata_df.loc[idx, "egfp_threshold"] = float(threshold)
        metadata_df.loc[idx, "egfp_reviewed"] = True
    else:
        metadata_df.loc[len(metadata_df)] = _normalize_metadata_record({
            "filename": opened_file,
            "egfp_threshold": float(threshold),
            "egfp_reviewed": True,
        })
    _persist_metadata()
    refresh_gfp_review_controls()
    if dpg.does_item_exist("gfp_status_text"):
        dpg.set_value("gfp_status_text", f"GFP Status: Accepted threshold {float(threshold):.2f}")

def make_plots_callback(sender=None, app_data=None, user_data=None):
    if not current_folder:
        if dpg.does_item_exist("plot_status_text"):
            dpg.set_value("plot_status_text", "Plot Status: Open a folder first.")
        return

    try:
        import postprocess_plots

        min_roundness = dpg.get_value("plot_min_roundness") if dpg.does_item_exist("plot_min_roundness") else None
        max_diameter_um = dpg.get_value("plot_max_diameter_um") if dpg.does_item_exist("plot_max_diameter_um") else None

        csv_path, output_dir = postprocess_plots.resolve_inputs(current_folder)
        df = pd.read_csv(csv_path)
        filtered_df = postprocess_plots.apply_filters(
            df,
            min_roundness=min_roundness,
            max_diameter_um=max_diameter_um,
        )
        postprocess_plots.cleanup_legacy_plot_files(output_dir)
        summaries = postprocess_plots.generate_plots(filtered_df, output_dir)

        if not summaries:
            raise RuntimeError("No plots were generated from the current processed CSV.")

        if dpg.does_item_exist("plot_status_text"):
            dpg.set_value(
                "plot_status_text",
                f"Plot Status: Saved {len(summaries)} plot(s) for {len(filtered_df)} filtered rows."
            )
        if dpg.does_item_exist("trace_status_text"):
            dpg.set_value("trace_status_text", f"Status: Saved plots to {output_dir}")
    except Exception as exc:
        if dpg.does_item_exist("plot_status_text"):
            dpg.set_value("plot_status_text", f"Plot Status: Error - {exc}")

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
    global current_folder, opened_file, channel_zstack, channel2_stack, channel_gfp_stack, opened_file_channel_indices
    current_folder = folder
    opened_file = None
    channel_zstack = None
    channel2_stack = None
    channel_gfp_stack = None
    opened_file_channel_indices = None
    _clear_gfp_review_state()
    _load_metadata_for_folder(folder)
    two_level = f"{os.path.basename(os.path.dirname(folder))}/{os.path.basename(folder)}"
    dpg.set_value("dir_path_repeat", two_level)
    refresh_contents_list()
    for tag in ("z_range_group", "rip_group", "wga_group", "gfp_review_window"):
        if dpg.does_item_exist(tag):
            dpg.hide_item(tag)
    refresh_gfp_review_controls()
    _update_gfp_status()
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
    global current_folder, opened_file, channel_zstack, channel2_stack, channel_gfp_stack, opened_file_channel_indices
    root = tk.Tk(); root.withdraw()
    folder = filedialog.askdirectory(); root.destroy()
    if not folder:
        return
    current_folder = folder
    opened_file = None
    channel_zstack = None
    channel2_stack = None
    channel_gfp_stack = None
    opened_file_channel_indices = None
    _clear_gfp_review_state()
    _load_metadata_for_folder(folder)
    two_level = f"{os.path.basename(os.path.dirname(folder))}\\{os.path.basename(folder)}"
    dpg.set_value("dir_path_repeat", two_level)
    refresh_contents_list()
    for tag in ("z_range_group","rip_group","wga_group","gfp_review_window"):
        if dpg.does_item_exist(tag):
            dpg.hide_item(tag)
    refresh_gfp_review_controls()
    _update_gfp_status()
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
    _rebuild_gfp_segmentation_regions()

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
        _rebuild_gfp_segmentation_regions()
    display_segmentation_filtered()
    if gfp_review_projection is not None and gfp_review_overlay_masks is not None:
        display_gfp_review()

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
        _rebuild_gfp_segmentation_regions()
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
    _rebuild_gfp_segmentation_regions()
    _clear_segmentation_render_cache()
    display_segmentation_filtered()
    if gfp_review_projection is not None and gfp_review_overlay_masks is not None:
        display_gfp_review()

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
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist("status_text"):
        return
    sel = _selected_filename_from_ui(app_data)
    if not sel:
        return
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
    global opened_file, channel_zstack, channel2_stack, channel_gfp_stack, opened_file_channel_indices
    global gray_img, mask_array, colors, selected_masks, texture_cache
    sel = _selected_filename_from_ui(app_data)
    if not sel:
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", "No file selected")
        return

    if sel != opened_file:
        dpg.set_value("status_text", f"Loading: {sel}")

        # Reset internal state
        opened_file_channel_indices = None
        channel_gfp_stack = None
        mask_array = None
        selected_masks.clear()
        colors.clear()
        texture_cache = None
        _clear_segmentation_render_cache()
        
        # Load image data
        path = os.path.join(current_folder, sel)
        with nd2.ND2File(path) as f:
            stack8 = to_8bit(f.asarray())
        channel_indices = get_image_channel_indices(stack8.shape[1])
        opened_file_channel_indices = dict(channel_indices)
        channel_zstack = stack8[:, channel_indices["dapi"], :, :]
        channel2_stack = stack8[:, channel_indices["wga"], :, :]
        channel_gfp_stack = stack8[:, channel_indices["egfp"], :, :] if channel_indices.get("egfp") is not None else None
        gray_img = max_proj(channel_zstack)
        opened_file = sel
        _restore_contents_selection()

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
        if channel_indices.get("egfp") is not None:
            load_gfp_review_if_available(sel)
        else:
            _clear_gfp_review_state()
            if dpg.does_item_exist("gfp_review_window"):
                dpg.hide_item("gfp_review_window")
            refresh_gfp_review_controls()
            _update_gfp_status()

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
    existing_egfp_threshold = np.nan
    existing_egfp_reviewed = False
    existing_z_min = None
    existing_z_max = None
    if opened_file in metadata_df["filename"].values:
        existing_row = metadata_df.loc[metadata_df["filename"] == opened_file].iloc[0]
        existing_rip_cells = existing_row["rip_cells"]
        existing_egfp_threshold = existing_row.get("egfp_threshold", np.nan)
        existing_egfp_reviewed = existing_row.get("egfp_reviewed", False)
        existing_z_min = int(existing_row.get("z_min", 0))
        existing_z_max = int(existing_row.get("z_max", 0))

    data = {
        "filename": opened_file,
        "z_min": z0,
        "z_max": z1,
        "rip_cells": _normalize_rip_cells(existing_rip_cells),
        "eye": eye,
        "time_min": time_min,
        "djid": djid,
        "treatment": treatment,
        "stain": stain,
        "egfp_threshold": existing_egfp_threshold,
        "egfp_reviewed": existing_egfp_reviewed,
    }

    if opened_file in metadata_df["filename"].values:
        idx = metadata_df["filename"] == opened_file
        metadata_df.loc[idx, :] = pd.DataFrame([data])
    else:
        metadata_df.loc[len(metadata_df)] = data

    _persist_metadata()
    refresh_contents_list()
    _restore_contents_selection()

    dpg.show_item("rip_group")
    z_range_changed = existing_z_min is not None and existing_z_max is not None and (existing_z_min != z0 or existing_z_max != z1)
    if z_range_changed and gfp_channel_exists_for_opened_file():
        invalidate_gfp_review_state(opened_file, remove_cache=True, clear_loaded=True, persist=True)
        dpg.set_value("status_text", f"Saved metadata for {opened_file}. GFP review reset because Z range changed.")
    else:
        dpg.set_value("status_text", f"Saved metadata for {opened_file}")

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
    img = gray_img if not dpg.get_value("wga_checkbox") or channel2_stack is None else channel2_stack[dpg.get_value("wga_slider")]
    update_texture(img, force=True)

def update_texture(base_img=None, force=False):
    global gray_img, channel2_stack, texture_cache, last_show_masks, last_selected, mask_array, selected_masks, colors
    print("\n[DEBUG] update_texture called")
    if base_img is None:
        base_img = gray_img if not dpg.get_value("wga_checkbox") or channel2_stack is None else channel2_stack[dpg.get_value("wga_slider")]

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
