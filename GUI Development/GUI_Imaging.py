import dearpygui.dearpygui as dpg
from fdialog import FileDialog
import os
import GUI_helpers as gh
from GUI_helpers import (
    refresh_contents_list,
    contents_list_callback,
    open_nd2_callback,
    z_slider_callback,
    run_rip_detector_callback,
    rip_checkbox_callback,
    wga_view_callback,
    update_texture,
    mask_click_callback,
    confirm_mask_selection_callback,
    save_metadata_callback
)
from analysis_helpers import segment_images, extract_traces

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
print("[DEBUG] CWD:", os.getcwd())

dpg.create_context()

def folder_selected_callback(paths):
    if not paths:
        return

    folder = paths[0]
    print(f"[DEBUG] Selected folder: {folder}")

    gh.current_folder = folder
    gh.opened_file = None

    two_level = f"{os.path.basename(os.path.dirname(folder))}/{os.path.basename(folder)}"
    dpg.set_value("dir_path_repeat", two_level)

    gh.refresh_contents_list()

    for tag in ("z_range_group", "rip_group", "wga_group"):
        if dpg.does_item_exist(tag):
            dpg.hide_item(tag)

    dpg.set_value("status_text", "Folder loaded")

folder_picker = FileDialog(
    callback=folder_selected_callback,
    dirs_only=True,
    default_path=".",
    modal=False,
    allow_drag=False
)

# Create texture registry with persistent dynamic textures
with dpg.texture_registry(show=False):
    import numpy as np
    # Main image texture: 1024x512
    _placeholder = np.zeros((512, 1024, 4), dtype=np.float32).flatten().tolist()
    dpg.add_dynamic_texture(1024, 512, _placeholder, tag="dynamic_texture")
    # Segmentation texture: match main image size (1024x512) so panels stay locked
    _placeholder_seg = np.zeros((512, 1024, 4), dtype=np.float32).flatten().tolist()
    dpg.add_dynamic_texture(1024, 512, _placeholder_seg, tag="segmentation_texture")

dpg.create_viewport(title='GUI', width=1430, height=1120)

with dpg.window(tag="left_window", label="Controls", pos=(10, 10), width=330, height=200):
    dpg.add_button(label="Open Folder", callback=folder_picker.show_file_dialog)
    dpg.add_text("None", tag="dir_path_repeat")
    dpg.add_checkbox(label="Save Meta Data", tag="opt_save_metadata", default_value=True)
    dpg.add_checkbox(label="Save Extracted Traces", tag="opt_save_traces", default_value=True)
    dpg.add_checkbox(label="Save Analyzed Data", tag="opt_save_analyzed", default_value=True)    
    dpg.add_text("Status: Ready", tag="status_text")

with dpg.window(tag="contents_window", label="Folder Contents", pos=(10, 220), width=330, height=505):
    dpg.add_listbox(items=[], tag="contents_list", num_items=10, width=315, callback=contents_list_callback)
    dpg.add_button(label="Open .nd2", width = 315, callback=open_nd2_callback)

    dpg.add_spacer(height=10)

    with dpg.group(tag="identifiers_group", show=False):
        with dpg.group(horizontal=True):
            dpg.add_text("DJID:")
            dpg.add_input_text(tag="djid_input", readonly=False, width=37)
            dpg.add_text("Eye:")
            dpg.add_combo(items=["L", "R", "Unknown"], tag="eye_combo", label="", width=30)
            dpg.add_text("Stain:")
            dpg.add_combo(items=["GLUT1", "GLUT3"], tag="stain_combo", label="", width=50)
        with dpg.group(horizontal=True):
            dpg.add_text("Treatment:")
            dpg.add_combo(items=["sutured", "open", "light flicker", "dark"], tag="treatment_combo", label="", width=100)
        with dpg.group(horizontal=True):
            dpg.add_text("Duration (min):")
            dpg.add_input_text(tag="time_input", hint="e.g. 0, 15, 30, 60, 90", width=201)

    dpg.add_spacer(height=10)

    with dpg.group():
        with dpg.group(tag="wga_group", show=False, horizontal=True):
            dpg.add_text("WGA View")
            dpg.add_checkbox(tag="wga_checkbox", callback=wga_view_callback)
            dpg.add_slider_int(tag="wga_slider", min_value=0, max_value=0, width = 223, callback=wga_view_callback)

        dpg.add_spacer(height=10)

with dpg.window(tag="rip_panel", label="Rip Panel", pos=(10, 735), width=330, height=125):
    with dpg.group(tag="rip_group", show=False):
        dpg.add_checkbox(label="Rip?", tag="rip_checkbox", callback=rip_checkbox_callback)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Rip Detector Mode", tag="run_rip_button", show=False, callback=run_rip_detector_callback)
            dpg.add_checkbox(label="Show Masks", tag="show_masks_checkbox", default_value=True, show=False, callback=lambda s, a, u: update_texture())
        dpg.add_text("Selected: 0", tag="selected_mask_count")
        dpg.add_button(label="Confirm Masks", tag="confirm_masks_button", show=False, callback=confirm_mask_selection_callback)

with dpg.window(tag="analysis_panel", label="Analysis Panel", pos=(10, 870), width=330, height=150):
    dpg.add_text("Step 1: Segmentation", color=(200, 200, 0))
    dpg.add_button(label="Segment Images", tag="segment_images_button", width=315, callback=segment_images)
    dpg.add_spacer(height=5)
    dpg.add_text("Step 2: Extract Traces", color=(200, 200, 0))
    dpg.add_button(label="Extract Traces", tag="extract_traces_button", width=315, callback=extract_traces)
    dpg.add_spacer(height=8)
    with dpg.group(tag="seg_options_group"):
        dpg.add_text("Segmentation Options", color=(180,180,255))
        dpg.add_checkbox(label="Show Removed Masks (grey)", tag="show_removed_masks", default_value=True, callback=lambda s,a,u: gh.display_segmentation_filtered())
        dpg.add_checkbox(label="Label Filtered Masks Only", tag="label_filtered_only", default_value=True, callback=lambda s,a,u: gh.display_segmentation_filtered())
        dpg.add_checkbox(label="Overlay On Image", tag="seg_overlay_on_image", default_value=False, callback=lambda s,a,u: gh.display_segmentation_filtered())
        dpg.add_slider_float(label="Overlay Alpha", tag="seg_overlay_alpha", min_value=0.0, max_value=1.0, default_value=0.45, width=160, callback=lambda s,a,u: gh.display_segmentation_filtered())
        dpg.add_checkbox(label="Manual Filter Mode", tag="manual_filter_mode", default_value=False)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply Manual Filter", tag="apply_manual_filter_button", callback=gh.apply_manual_filter, width=150)
            dpg.add_button(label="Reset Manual Filter", tag="reset_manual_filter_button", callback=gh.reset_manual_filter, width=150)
    dpg.add_text("File: None", tag="trace_file_status", wrap=280)
    dpg.add_text("Status: Waiting", tag="trace_status_text", wrap=280)


with dpg.window(tag="right_window", label="Image Panel", pos=(350, 10), width=1070, height=600):
    with dpg.drawlist(tag="drawlist", width=1024, height=512):
        dpg.draw_image("dynamic_texture", (0, 0), (1024, 512))
    handler = dpg.add_item_handler_registry()
    dpg.add_item_clicked_handler(callback=mask_click_callback, parent=handler)
    dpg.bind_item_handler_registry("drawlist", handler)

with dpg.window(tag="segmentation_window", label="Segmentation", pos=(350, 620), width=1070, height=540):
    with dpg.drawlist(tag="segmentation_drawlist", width=1024, height=512):
        dpg.draw_image("segmentation_texture", (0, 0), (1024, 512))
    seg_handler = dpg.add_item_handler_registry()
    dpg.add_item_clicked_handler(callback=gh.segmentation_click_callback, parent=seg_handler)
    dpg.bind_item_handler_registry("segmentation_drawlist", seg_handler)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
