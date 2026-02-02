#!/usr/bin/env python
"""Fix color assignment saving and loading"""
import re

# Fix analysis_helpers.py
print("Fixing analysis_helpers.py...")
with open('analysis_helpers.py', 'r') as f:
    lines = f.readlines()

# Insert color assignment before savez (around line 611)
# Find the line with "color_assignment = assign_colors_to_masks"
for i, line in enumerate(lines):
    if 'color_assignment = assign_colors_to_masks(dapi_masks)' in line and 'Save segmentation results' in ''.join(lines[i-15:i]):
        # This is the OLD location (after savez). Move it before
        # Find the savez start
        for j in range(i-1, max(0, i-30), -1):
            if 'np.savez(output_file,' in lines[j]:
                # Found savez, go back to find where to insert
                for k in range(j-1, max(0, j-20), -1):
                    if 'mask_ids = np.delete' in lines[k]:
                        # Insert after this line
                        insert_lines = [
                            '\n',
                            '        # Assign colors BEFORE saving\n',
                            '        color_assignment = assign_colors_to_masks(dapi_masks)\n'
                        ]
                        lines = lines[:k+1] + insert_lines + lines[k+1:]
                        # Now remove the old location
                        # Find and remove the old color_assignment line
                        for m in range(len(lines)):
                            if m > k+4 and 'color_assignment = assign_colors_to_masks(dapi_masks)' in lines[m]:
                                lines.pop(m)
                                break
                        break
                break
        break

# Add color_assignment to savez call
for i, line in enumerate(lines):
    if 'filtered_idxs=filtered_idxs,' in line and 'stack=stack,' in lines[i+1]:
        # Insert color_assignment
        lines.insert(i+1, '                 color_assignment=color_assignment,\n')
        break

with open('analysis_helpers.py', 'w') as f:
    f.writelines(lines)
print("✓ Fixed analysis_helpers.py")

# Fix GUI_helpers.py
print("Fixing GUI_helpers.py...")
with open('GUI_helpers.py', 'r') as f:
    content = f.read()

# Replace the part where we assign colors
old_pattern = r"segmentation_colors = assign_colors_to_masks\(segmentation_masks\)"
new_code = """# Load color assignment from npz if available
                if 'color_assignment' in seg_data:
                    color_dict = seg_data['color_assignment'].item()  # Convert numpy array wrapper to dict
                    segmentation_colors = color_dict
                    print(f"[DEBUG] Loaded {len(segmentation_colors)} colors from npz")
                else:
                    import analysis_helpers
                    segmentation_colors = analysis_helpers.assign_colors_to_masks(segmentation_masks)
                    print(f"[DEBUG] Computed {len(segmentation_colors)} colors from masks")"""

content = re.sub(
    r"segmentation_colors = assign_colors_to_masks\(segmentation_masks\)",
    new_code,
    content
)

with open('GUI_helpers.py', 'w') as f:
    f.write(content)
print("✓ Fixed GUI_helpers.py")

print("\nAll fixes applied!")
