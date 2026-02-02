#!/usr/bin/env python
# Temporary script to fix the color assignment

# Read the file
with open('analysis_helpers.py', 'r') as f:
    content = f.read()

# Find and replace the section
old_section = """        mask_ids = np.delete(np.unique(dapi_masks), 0) - 1

        # Save segmentation results for each stack
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(segmentation_dir, f"{base_name}_segmentation.npz")
        
        np.savez(output_file,
                 dapi_masks=dapi_masks,
                 filtered_idxs=filtered_idxs,
                 stack=stack,
                 cropped_stack=cropped_stack,
                 dapi_stack=dapi_stack,
                 filename=filename,
                 z_min=z_min,
                 z_max=z_max)
        
        print(f"Saved segmentation to: {output_file}")
        
        # Assign colors to maximize visual differences between adjacent masks
        color_assignment = assign_colors_to_masks(dapi_masks)"""

new_section = """        mask_ids = np.delete(np.unique(dapi_masks), 0) - 1
        
        # Assign colors to maximize visual differences between adjacent masks
        color_assignment = assign_colors_to_masks(dapi_masks)

        # Save segmentation results for each stack
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(segmentation_dir, f"{base_name}_segmentation.npz")
        
        np.savez(output_file,
                 dapi_masks=dapi_masks,
                 filtered_idxs=filtered_idxs,
                 color_assignment=color_assignment,
                 stack=stack,
                 cropped_stack=cropped_stack,
                 dapi_stack=dapi_stack,
                 filename=filename,
                 z_min=z_min,
                 z_max=z_max)
        
        print(f"Saved segmentation to: {output_file}")"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('analysis_helpers.py', 'w') as f:
        f.write(content)
    print("✓ Updated segment_images to save color_assignment")
else:
    print("✗ Could not find target section in analysis_helpers.py")

# Now fix GUI_helpers to load the colors
with open('GUI_helpers.py', 'r') as f:
    gh_content = f.read()

old_gh = """    # Try to load segmentation
    segmentation_dir = os.path.join(folder_path, f"{folder_name}_segmentation")
    if os.path.exists(segmentation_dir):
        seg_files = [f for f in os.listdir(segmentation_dir) if f.endswith('_segmentation.npz')]
        if seg_files:
            seg_file = os.path.join(segmentation_dir, seg_files[0])
            try:
                seg_data = np.load(seg_file, allow_pickle=True)
                global segmentation_masks, segmentation_colors, segmentation_filtered_idxs
                segmentation_masks = seg_data['dapi_masks']
                segmentation_filtered_idxs = list(seg_data['filtered_idxs'])
                segmentation_colors = assign_colors_to_masks(segmentation_masks)
                print(f"[DEBUG] Loaded segmentation: {len(segmentation_masks)} masks, {len(segmentation_filtered_idxs)} filtered")
                display_segmentation_filtered()
            except Exception as e:
                print(f"[ERROR] Failed to load segmentation: {e}")
                import traceback
                traceback.print_exc()"""

new_gh = """    # Try to load segmentation
    segmentation_dir = os.path.join(folder_path, f"{folder_name}_segmentation")
    if os.path.exists(segmentation_dir):
        seg_files = [f for f in os.listdir(segmentation_dir) if f.endswith('_segmentation.npz')]
        if seg_files:
            seg_file = os.path.join(segmentation_dir, seg_files[0])
            try:
                seg_data = np.load(seg_file, allow_pickle=True)
                global segmentation_masks, segmentation_colors, segmentation_filtered_idxs
                segmentation_masks = seg_data['dapi_masks']
                segmentation_filtered_idxs = list(seg_data['filtered_idxs'])
                # Load color assignment from npz if available
                if 'color_assignment' in seg_data:
                    color_dict = seg_data['color_assignment'].item()  # Convert numpy array wrapper to dict
                    segmentation_colors = color_dict
                    print(f"[DEBUG] Loaded {len(segmentation_colors)} colors from npz")
                else:
                    import analysis_helpers
                    segmentation_colors = analysis_helpers.assign_colors_to_masks(segmentation_masks)
                    print(f"[DEBUG] Computed {len(segmentation_colors)} colors from masks")
                print(f"[DEBUG] Loaded segmentation: {len(segmentation_masks)} masks, {len(segmentation_filtered_idxs)} filtered")
                display_segmentation_filtered()
            except Exception as e:
                print(f"[ERROR] Failed to load segmentation: {e}")
                import traceback
                traceback.print_exc()"""

if old_gh in gh_content:
    gh_content = gh_content.replace(old_gh, new_gh)
    with open('GUI_helpers.py', 'w') as f:
        f.write(gh_content)
    print("✓ Updated GUI_helpers to load color_assignment from npz")
else:
    print("✗ Could not find load_segmentation_if_available in GUI_helpers.py")

print("\nDone!")
