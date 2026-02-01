import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import dearpygui.dearpygui as dpg
from skimage import img_as_ubyte
import nd2
from dearpygui_obj.userwidget import UserWidget
import cv2

# Helper: convert to 8-bit
def to_8bit(arr):
    arrf = arr.astype(np.float32)
    maxv = arrf.max() if arrf.size else 0.0
    if maxv > 0:
        norm = arrf / maxv
    else:
        norm = arrf * 0.0
    return img_as_ubyte(norm)

# Max projection helper
def max_proj(channel_zstack):
    return np.max(channel_zstack, axis=0)

# Custom Z‑range slider widget
def z_slider_callback(sender, app_data, user_data):
    """
    Receives app_data as (min_z, max_z) and updates the projection.
    """
    global gray_img
    if channel_zstack is None:
        return
    z0, z1 = app_data
    proj = max_proj(channel_zstack[z0:z1+1])
    gray_img = proj
    update_texture()

class ZRangeSlider(UserWidget):
    def __init__(self, tag, parent, max_slice, callback, default_min=0):
        self.max_slice = max_slice
        self.callback = callback
        self.default_min = default_min
        # default_max at end of stack
        self.default_max = max_slice
        super().__init__(tag=tag, parent=parent)

    def __setup_content__(self):
        dpg.add_text("Z Range:")
        self.min_tag = f"{self.tag}_min"
        self.max_tag = f"{self.tag}_max"
        dpg.add_slider_int(
            label="Min Z",
            tag=self.min_tag,
            min_value=0,
            max_value=self.max_slice,
            default_value=self.default_min,
            callback=self._on_change,
            width=280
        )
        dpg.add_slider_int(
            label="Max Z",
            tag=self.max_tag,
            min_value=0,
            max_value=self.max_slice,
            default_value=self.default_max,
            callback=self._on_change,
            width=280
        )

    def _on_change(self, sender, app_data, user_data):
        # Clamp sliders
        mn = dpg.get_value(self.min_tag)
        mx = dpg.get_value(self.max_tag)
        if mx < mn:
            dpg.set_value(self.max_tag, mn)
            mx = mn
        if mn > mx:
            dpg.set_value(self.min_tag, mx)
            mn = mx
        # callback with (min, max)
        self.callback(None, (mn, mx), None)

# Globals for current data
current_folder = None
mask_array = None
gray_img = None
channel_zstack = None
colors = {}
selected_masks = []

# Fixed dynamic texture canvas size
_TEXTURE_SIZE = 2048

def _set_dynamic_texture_from_array(rgba):
    """
    Put `rgba` into the central region of a single fixed-size dynamic texture
    and update it via `dpg.set_value`. Rescales if image is larger than the canvas.
    """
    if rgba is None:
        return
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("rgba must be HxWx4")

    h, w = arr.shape[0], arr.shape[1]

    # Downscale if needed to fit in canvas while preserving aspect ratio
    max_side = _TEXTURE_SIZE
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        arr_resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        arr_resized = arr
        new_h, new_w = h, w

    # Place resized image centered in canvas
    canvas = np.zeros((_TEXTURE_SIZE, _TEXTURE_SIZE, 4), dtype=np.float32)
    y0 = (_TEXTURE_SIZE - new_h) // 2
    x0 = (_TEXTURE_SIZE - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w, :] = arr_resized

    # Flatten and update the single texture
    flat = canvas.flatten().tolist()
    try:
        dpg.set_value("dynamic_texture", flat)
    except Exception as exc:
        print(f"[WARN] set_value failed: {exc}")

# Blend function: masks at 10% alpha, selected masks fully opaque
def blend_with_masks(gray, masks, colors, selected, alpha=0.1):
    base = gray.astype(np.float32)
    if base.max() > 1.0:
        base /= 255.0
    h, w = base.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = np.stack([base] * 3, axis=-1)
    rgba[..., 3] = 1.0
    if masks is not None:
        for m, col in colors.items():
            sel = (masks == m)
            rgba[sel, :3] = col
            rgba[sel, 3] = 1.0 if m in selected else alpha
    return rgba

# Initialize DearPyGui and placeholder texture
dpg.create_context()
with dpg.texture_registry(show=False):
    placeholder = np.ones((1, 1, 4), dtype=np.float32).flatten().tolist()
    dpg.add_dynamic_texture(1, 1, placeholder, tag="dynamic_texture")

# Core callbacks
def open_folder_dialog(sender, app_data, user_data):
    global current_folder
    root = tk.Tk(); root.withdraw()
    folder = filedialog.askdirectory(); root.destroy()
    if not folder:
        dpg.set_value("dir_path", "None")
        dpg.set_value("dir_path_repeat", "None")
        dpg.configure_item("contents_list", items=[])
        return
    current_folder = folder
    parent = os.path.basename(folder)
    grandparent = os.path.basename(os.path.dirname(folder))
    two_level = f"{grandparent}\\{parent}"
    dpg.set_value("dir_path", two_level)
    dpg.set_value("dir_path_repeat", two_level)
    nd2_files = [f for f in os.listdir(folder) if f.lower().endswith('.nd2')]
    dpg.configure_item("contents_list", items=nd2_files)
    dpg.set_value("status_text", "Folder loaded")

def open_nd2_callback(sender, app_data, user_data):
    global gray_img, mask_array, channel_zstack, colors, selected_masks
    filename = dpg.get_value("contents_list")
    if not current_folder or not filename:
        return
    dpg.set_value("status_text", f"Loading: {filename}")
    f = nd2.ND2File(os.path.join(current_folder, filename))
    stack = to_8bit(f.asarray()); f.close()
    channel_zstack = stack[:, 0, :, :]
    depth = channel_zstack.shape[0]
    # Remove any existing widget
    if dpg.does_item_exist("zrange_widget"):
        dpg.delete_item("zrange_widget")
    # Instantiate custom ZRangeSlider in Contents pane
    ZRangeSlider(tag="zrange_widget",
                 parent="contents_window",
                 max_slice=depth-1,
                 callback=z_slider_callback)
    # Initial image projection
    proj = max_proj(channel_zstack)
    gray_img = proj
    mask_array = np.zeros_like(proj, dtype=int)
    colors.clear(); selected_masks.clear()
    h, w = proj.shape
    data = blend_with_masks(proj, mask_array, colors, selected_masks)
    _set_dynamic_texture_from_array(data)
    dpg.configure_item("drawlist", width=w, height=h)
    dpg.configure_item("right_window", width=w+20, height=h+200)
    dpg.delete_item("drawlist", children_only=True)
    dpg.draw_image("dynamic_texture", (0,0), (w,h), parent="drawlist")
    dpg.set_value("nd2_status", f"Loaded: {filename}")
    dpg.set_value("status_text", f"Loaded: {filename}")

# Mask click
def mask_click_callback(sender, app_data, user_data):
    if mask_array is None or mask_array.max() == 0:
        return
    mx, my = dpg.get_mouse_pos(local=False)
    x0, y0 = dpg.get_item_rect_min("drawlist")
    ix, iy = int(mx-x0), int(my-y0)
    if ix<0 or iy<0 or ix>=gray_img.shape[1] or iy>=gray_img.shape[0]:
        return
    m_id = int(mask_array[iy,ix])
    if m_id>0:
        if m_id in selected_masks: selected_masks.remove(m_id)
        else: selected_masks.append(m_id)
        dpg.configure_item("selected_list", items=[f"Mask {m}" for m in selected_masks])
        update_texture()
    dpg.set_value("status_text", f"Selected: {m_id}")

# Update texture
def update_texture():
    data = blend_with_masks(gray_img, mask_array, colors, selected_masks)
    _set_dynamic_texture_from_array(data)

# Viewport resize
def viewport_resize_callback(sender, app_data):
    width = dpg.get_viewport_width(); left=300
    dpg.configure_item("left_window", width=left)
    dpg.configure_item("contents_window", width=left)
    dpg.configure_item("right_window", pos=(left+20,10), width=width-(left+30))

# Build UI
dpg.create_viewport(title='Masked Image GUI', width=1400, height=1100)
with dpg.window(tag="left_window", label="Controls", pos=(10,10), width=300, height=300, no_move=True):
    dpg.add_button(label="Open Folder", callback=open_folder_dialog, width=-1)
    dpg.add_text("Selected Folder:")
    dpg.add_text("None", tag="dir_path", wrap=280)
    dpg.add_spacer(height=5)
    dpg.add_checkbox(label="eGFP cells", tag="opt_egfp", default_value=True)
    dpg.add_checkbox(label="WGA rips", tag="opt_wga", default_value=False)
    dpg.add_checkbox(label="Save MP details", tag="opt_save", default_value=True)
    dpg.add_text("Status: Ready", tag="status_text")
with dpg.window(tag="contents_window", label="Folder Contents", pos=(10,320), width=300, height=300, no_move=True):
    dpg.add_text("Selected Folder:")
    dpg.add_text("None", tag="dir_path_repeat", wrap=280)
    dpg.add_listbox(items=[], tag="contents_list", num_items=5, width=280)
    dpg.add_button(label="Open .nd2", callback=open_nd2_callback, width=-1)
    dpg.add_text("No ND2 loaded", tag="nd2_status")
with dpg.window(tag="right_window", label="Image Panel", pos=(355,10), width=1024, height=1024, no_move=True):
    dpg.add_drawlist(tag="drawlist", width=1024, height=1024)
    dpg.draw_image("dynamic_texture", (0,0), (1024,1024), parent="drawlist")
    reg=dpg.add_item_handler_registry()
    dpg.add_item_clicked_handler(callback=mask_click_callback, parent=reg)
    dpg.bind_item_handler_registry("drawlist", reg)
    dpg.add_listbox(items=[], tag="selected_list", num_items=15, width=300)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()