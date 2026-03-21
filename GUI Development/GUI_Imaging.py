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
    save_metadata_callback,
    make_plots_callback
)
from analysis_helpers import segment_images, extract_traces

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
print("[DEBUG] CWD:", os.getcwd())

dpg.create_context()

DEFAULT_VIEWPORT_WIDTH = 1560
DEFAULT_VIEWPORT_HEIGHT = 1320

def folder_selected_callback(paths):
    if not paths:
        return

    folder = paths[0]
    print(f"[DEBUG] Selected folder: {folder}")
    gh.handle_folder_selection(folder)

folder_picker = FileDialog(
    callback=folder_selected_callback,
    dirs_only=True,
    default_path=".",
    modal=False,
    allow_drag=False,
    show_shortcuts_menu=False
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

dpg.create_viewport(title='GUI', width=DEFAULT_VIEWPORT_WIDTH, height=DEFAULT_VIEWPORT_HEIGHT)

with dpg.window(tag="left_window", label="Controls", pos=(10, 10), width=390, height=190):
    dpg.add_button(label="Open Folder", tag="open_folder_button", width=-1, callback=folder_picker.show_file_dialog)
    dpg.add_text("None", tag="dir_path_repeat", wrap=360)
    dpg.add_checkbox(label="Has GFP Channel", tag="has_gfp_channel", default_value=True, callback=gh.gfp_channel_toggle_callback)
    dpg.add_checkbox(label="Save Meta Data", tag="opt_save_metadata", default_value=True)
    dpg.add_checkbox(label="Save Extracted Traces", tag="opt_save_traces", default_value=True)
    dpg.add_checkbox(label="Save Analyzed Data", tag="opt_save_analyzed", default_value=True)    
    dpg.add_text("Status: Ready", tag="status_text", wrap=360)

with dpg.window(tag="contents_window", label="Folder Contents", pos=(10, 210), width=390, height=560):
    dpg.add_listbox(items=[], tag="contents_list", num_items=10, width=-1, callback=contents_list_callback)
    dpg.add_button(label="Open .nd2", tag="open_nd2_button", width=-1, callback=open_nd2_callback)

    dpg.add_spacer(height=10)

    with dpg.group(tag="identifiers_group", show=False):
        with dpg.group(horizontal=True):
            dpg.add_text("DJID:")
            dpg.add_input_text(tag="djid_input", readonly=False, width=90)
            dpg.add_text("Eye:")
            dpg.add_combo(items=["L", "R", "Unknown"], tag="eye_combo", label="", width=95)
        with dpg.group(horizontal=True):
            dpg.add_text("Stain:")
            dpg.add_combo(items=["GLUT1", "GLUT3"], tag="stain_combo", label="", width=180)
        with dpg.group(horizontal=True):
            dpg.add_text("Treatment:")
            dpg.add_combo(items=["sutured", "open", "light flicker", "dark"], tag="treatment_combo", label="", width=180)
        with dpg.group(horizontal=True):
            dpg.add_text("Duration (min):")
            dpg.add_input_text(tag="time_input", hint="e.g. 0, 15, 30, 60, 90", width=180)

    dpg.add_spacer(height=10)

    with dpg.group():
        with dpg.group(tag="wga_group", show=False, horizontal=True):
            dpg.add_text("WGA View")
            dpg.add_checkbox(tag="wga_checkbox", callback=wga_view_callback)
            dpg.add_slider_int(tag="wga_slider", min_value=0, max_value=0, width = 223, callback=wga_view_callback)

        dpg.add_spacer(height=10)

with dpg.window(tag="rip_panel", label="Rip Panel", pos=(10, 780), width=390, height=125):
    with dpg.group(tag="rip_group", show=False):
        dpg.add_checkbox(label="Rip?", tag="rip_checkbox", callback=rip_checkbox_callback)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Rip Detector Mode", tag="run_rip_button", show=False, callback=run_rip_detector_callback)
            dpg.add_checkbox(label="Show Masks", tag="show_masks_checkbox", default_value=True, show=False, callback=lambda s, a, u: update_texture())
        dpg.add_text("Selected: 0", tag="selected_mask_count")
        dpg.add_button(label="Confirm Masks", tag="confirm_masks_button", show=False, callback=confirm_mask_selection_callback)

with dpg.window(tag="analysis_panel", label="Analysis Panel", pos=(10, 915), width=390, height=370):
    dpg.add_text("Step 1: Segmentation", color=(200, 200, 0))
    dpg.add_button(label="Segment Images", tag="segment_images_button", width=-1, callback=segment_images)
    dpg.add_checkbox(label="Recompute Existing Segmentations", tag="recompute_segmentation", default_value=False)
    dpg.add_spacer(height=5)
    dpg.add_text("Step 2: Extract Traces", color=(200, 200, 0))
    dpg.add_button(label="Extract Traces", tag="extract_traces_button", width=-1, callback=extract_traces)
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
    dpg.add_spacer(height=8)
    with dpg.group(tag="plot_options_group"):
        dpg.add_text("Step 3: Post-Processing Plots", color=(200, 200, 0))
        with dpg.group(horizontal=True):
            dpg.add_text("Min Roundness:")
            dpg.add_input_float(tag="plot_min_roundness", default_value=0.65, width=110, step=0.05, format="%.2f")
        with dpg.group(horizontal=True):
            dpg.add_text("Max Diam (um):")
            dpg.add_input_float(tag="plot_max_diameter_um", default_value=25.0, width=110, step=1.0, format="%.1f")
        dpg.add_button(label="Make Plots", tag="make_plots_button", width=-1, callback=make_plots_callback)
        dpg.add_text("Plot Status: Waiting", tag="plot_status_text", wrap=360)
    dpg.add_text("File: None", tag="trace_file_status", wrap=360)
    dpg.add_text("Status: Waiting", tag="trace_status_text", wrap=360)


with dpg.window(tag="right_window", label="Image Panel", pos=(420, 10), width=980, height=540):
    with dpg.drawlist(tag="drawlist", width=930, height=460):
        dpg.draw_image("dynamic_texture", (0, 0), (930, 460), tag="main_image_draw")
    handler = dpg.add_item_handler_registry()
    dpg.add_item_clicked_handler(callback=mask_click_callback, parent=handler)
    dpg.bind_item_handler_registry("drawlist", handler)

with dpg.window(tag="segmentation_window", label="Segmentation", pos=(420, 565), width=980, height=500):
    with dpg.drawlist(tag="segmentation_drawlist", width=930, height=420):
        dpg.draw_image("segmentation_texture", (0, 0), (930, 420), tag="segmentation_image_draw")
    seg_handler = dpg.add_item_handler_registry()
    dpg.add_item_clicked_handler(callback=gh.segmentation_click_callback, parent=seg_handler)
    dpg.bind_item_handler_registry("segmentation_drawlist", seg_handler)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
