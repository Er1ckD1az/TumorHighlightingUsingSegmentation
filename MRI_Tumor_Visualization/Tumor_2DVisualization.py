import numpy as np
import nibabel as nib
from skimage import filters, measure, morphology
import matplotlib.pyplot as plt
import os

def load_mri_data(file_path):
    # Load the NIfTI file
    img = nib.load(file_path)
    # Get data and affine transformation
    data = img.get_fdata()
    affine = img.affine
    
    return data, affine

def load_tumor_mask(mask_file_path):
    # Load the NIfTI file
    mask_img = nib.load(mask_file_path)
    
    # Get mask data and convert to boolean
    mask_data = mask_img.get_fdata()
    binary_mask = mask_data > 0
    
    return binary_mask, mask_img.affine

def segment_tumor(mri_data, threshold_factor=2.0):

    # Normalize the data
    normalized_data = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
    
    # Compute the threshold using Otsu's method and adjust with the factor
    threshold = filters.threshold_otsu(normalized_data) * threshold_factor
    
    # Create binary mask
    tumor_mask = normalized_data > threshold
    
    # Apply morphological operations to clean up the mask
    tumor_mask = morphology.remove_small_objects(tumor_mask, min_size=100)
    tumor_mask = morphology.binary_closing(tumor_mask, morphology.ball(2))
    
    # Label connected components
    labeled_mask, num_features = measure.label(tumor_mask, return_num=True)
    
    # Find the largest component (assuming it's the tumor)
    if num_features > 1:
        # Count pixels in each component
        component_sizes = np.bincount(labeled_mask.ravel())[1:]
        # Keep only the largest component (excluding background which is 0)
        largest_component = np.argmax(component_sizes) + 1
        tumor_mask = labeled_mask == largest_component
    
    return tumor_mask

def display_mri_slices(mri_data, tumor_mask, num_slices=5):
    # Find slices with tumor present
    tumor_slice_sizes = [np.sum(tumor_mask[:, :, i]) for i in range(tumor_mask.shape[2])]
    
    # Find indices of slices with tumors, sorted by tumor size
    tumor_slices = [(i, size) for i, size in enumerate(tumor_slice_sizes) if size > 0]
    tumor_slices.sort(key=lambda x: x[1], reverse=True)
    
    # If no tumor found in any slice, use middle slices
    if not tumor_slices:
        slice_indices = [mri_data.shape[2] // 2]
        num_slices = 1
        print("No tumor detected in any slice, showing middle slice.")
    else:
        # Use the slices with the largest tumor areas
        slice_indices = [idx for idx, _ in tumor_slices[:min(num_slices, len(tumor_slices))]]
    
    # Create a figure with multiple slices
    fig, axes = plt.subplots(2, len(slice_indices), figsize=(4*len(slice_indices), 8))
    
    # Handle case where only one slice is shown
    if len(slice_indices) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    # Display each slice
    for i, slice_idx in enumerate(slice_indices):
        # Show the MRI slice
        axes[0, i].imshow(mri_data[:, :, slice_idx], cmap='gray')
        axes[0, i].set_title(f'MRI Slice {slice_idx}')
        axes[0, i].axis('off')
        
        # Show the MRI with tumor overlay
        axes[1, i].imshow(mri_data[:, :, slice_idx], cmap='gray')
        
        # Create a red mask for the tumor
        mask = np.zeros((*mri_data[:, :, slice_idx].shape, 4))
        mask[:, :, 0] = 1.0  # Red channel
        mask[:, :, 3] = tumor_mask[:, :, slice_idx] * 0.7  # Alpha channel
        
        axes[1, i].imshow(mask)
        axes[1, i].set_title(f'Tumor Overlay (Slice {slice_idx})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def display_multi_view(mri_data, tumor_mask):
    # Find slices with the largest tumor cross-section in each orientation
    tumor_slice_sizes_z = [np.sum(tumor_mask[:, :, i]) for i in range(tumor_mask.shape[2])]
    tumor_slice_sizes_y = [np.sum(tumor_mask[:, i, :]) for i in range(tumor_mask.shape[1])]
    tumor_slice_sizes_x = [np.sum(tumor_mask[i, :, :]) for i in range(tumor_mask.shape[0])]
    
    # Get the middle slices as default
    mid_z = mri_data.shape[2] // 2
    mid_y = mri_data.shape[1] // 2
    mid_x = mri_data.shape[0] // 2
    
    # Find best slices (with largest tumor area)
    best_z = np.argmax(tumor_slice_sizes_z) if max(tumor_slice_sizes_z) > 0 else mid_z
    best_y = np.argmax(tumor_slice_sizes_y) if max(tumor_slice_sizes_y) > 0 else mid_y
    best_x = np.argmax(tumor_slice_sizes_x) if max(tumor_slice_sizes_x) > 0 else mid_x
    
    # Create a figure with three views
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Axial view (z-axis)
    axes[0, 0].imshow(mri_data[:, :, best_z], cmap='gray')
    axes[0, 0].set_title(f'Axial (z={best_z})')
    axes[0, 0].axis('off')
    
    # Axial view with tumor overlay
    axes[1, 0].imshow(mri_data[:, :, best_z], cmap='gray')
    mask_z = np.zeros((*mri_data[:, :, best_z].shape, 4))
    mask_z[:, :, 0] = 1.0  # Red channel
    mask_z[:, :, 3] = tumor_mask[:, :, best_z] * 0.7  # Alpha channel
    axes[1, 0].imshow(mask_z)
    axes[1, 0].set_title('Tumor Overlay (Axial)')
    axes[1, 0].axis('off')
    
    # Coronal view (y-axis)
    axes[0, 1].imshow(mri_data[:, best_y, :].T, cmap='gray')
    axes[0, 1].set_title(f'Coronal (y={best_y})')
    axes[0, 1].axis('off')
    
    # Coronal view with tumor overlay
    axes[1, 1].imshow(mri_data[:, best_y, :].T, cmap='gray')
    mask_y = np.zeros((*mri_data[:, best_y, :].T.shape, 4))
    mask_y[:, :, 0] = 1.0  # Red channel
    mask_y[:, :, 3] = tumor_mask[:, best_y, :].T * 0.7  # Alpha channel
    axes[1, 1].imshow(mask_y)
    axes[1, 1].set_title('Tumor Overlay (Coronal)')
    axes[1, 1].axis('off')
    
    # Sagittal view (x-axis)
    axes[0, 2].imshow(mri_data[best_x, :, :].T, cmap='gray')
    axes[0, 2].set_title(f'Sagittal (x={best_x})')
    axes[0, 2].axis('off')
    
    # Sagittal view with tumor overlay
    axes[1, 2].imshow(mri_data[best_x, :, :].T, cmap='gray')
    mask_x = np.zeros((*mri_data[best_x, :, :].T.shape, 4))
    mask_x[:, :, 0] = 1.0  # Red channel
    mask_x[:, :, 3] = tumor_mask[best_x, :, :].T * 0.7  # Alpha channel
    axes[1, 2].imshow(mask_x)
    axes[1, 2].set_title('Tumor Overlay (Sagittal)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_tumor_statistics(tumor_mask, voxel_dimensions=(1.0, 1.0, 1.0)):
    # Count voxels
    tumor_voxel_count = np.sum(tumor_mask)
    
    # Calculate volume in cubic mm (assuming voxel_dimensions are in mm)
    voxel_volume = np.prod(voxel_dimensions)
    tumor_volume_mm3 = tumor_voxel_count * voxel_volume
    
    # Convert to cubic cm
    tumor_volume_cm3 = tumor_volume_mm3 / 1000.0
    
    # Find tumor bounds
    if tumor_voxel_count > 0:
        z_indices, y_indices, x_indices = np.where(tumor_mask)
        
        # Calculate dimensions
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        z_min, z_max = np.min(z_indices), np.max(z_indices)
        
        # Calculate dimensions in mm
        x_dim = (x_max - x_min + 1) * voxel_dimensions[0]
        y_dim = (y_max - y_min + 1) * voxel_dimensions[1]
        z_dim = (z_max - z_min + 1) * voxel_dimensions[2]
        
        # Calculate center of mass
        com_x = np.mean(x_indices) * voxel_dimensions[0]
        com_y = np.mean(y_indices) * voxel_dimensions[1]
        com_z = np.mean(z_indices) * voxel_dimensions[2]
        
        return {
            "voxel_count": tumor_voxel_count,
            "volume_mm3": tumor_volume_mm3,
            "volume_cm3": tumor_volume_cm3,
            "dimensions_mm": (x_dim, y_dim, z_dim),
            "dimensions_voxels": (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1),
            "bounds_voxels": ((x_min, x_max), (y_min, y_max), (z_min, z_max)),
            "center_of_mass_mm": (com_x, com_y, com_z)
        }
    else:
        return {
            "voxel_count": 0,
            "volume_mm3": 0,
            "volume_cm3": 0,
            "dimensions_mm": (0, 0, 0),
            "dimensions_voxels": (0, 0, 0),
            "bounds_voxels": ((0, 0), (0, 0), (0, 0)),
            "center_of_mass_mm": (0, 0, 0)
        }

def save_results(mri_data, tumor_mask, output_dir, stats):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tumor statistics to a text file
    with open(os.path.join(output_dir, 'tumor_statistics.txt'), 'w') as f:
        f.write("TUMOR STATISTICS\n")
        f.write("================\n\n")
        f.write(f"Tumor volume: {stats['volume_cm3']:.2f} cm³ ({stats['voxel_count']} voxels)\n")
        f.write(f"Dimensions (mm): {stats['dimensions_mm'][0]:.2f} × {stats['dimensions_mm'][1]:.2f} × {stats['dimensions_mm'][2]:.2f}\n")
        f.write(f"Dimensions (voxels): {stats['dimensions_voxels'][0]} × {stats['dimensions_voxels'][1]} × {stats['dimensions_voxels'][2]}\n")
        f.write(f"Center of mass (mm): ({stats['center_of_mass_mm'][0]:.2f}, {stats['center_of_mass_mm'][1]:.2f}, {stats['center_of_mass_mm'][2]:.2f})\n")
    
    # Create and save visualizations
    # Find the slice with the largest tumor cross-section
    tumor_slice_sizes = [np.sum(tumor_mask[:, :, i]) for i in range(tumor_mask.shape[2])]
    best_slice = np.argmax(tumor_slice_sizes) if max(tumor_slice_sizes) > 0 else tumor_mask.shape[2] // 2
    
    # Save 2D visualizations
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show the MRI slice
    axes[0].imshow(mri_data[:, :, best_slice], cmap='gray')
    axes[0].set_title('MRI Slice')
    axes[0].axis('off')
    
    # Show the MRI with tumor overlay
    axes[1].imshow(mri_data[:, :, best_slice], cmap='gray')
    
    # Create a red mask for the tumor
    masked_data = np.zeros((*mri_data[:, :, best_slice].shape, 4))
    masked_data[:, :, 0] = 1.0  # Red channel
    masked_data[:, :, 3] = tumor_mask[:, :, best_slice] * 0.7  # Alpha channel
    
    axes[1].imshow(masked_data)
    axes[1].set_title('Tumor Overlay')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tumor_visualization_2d.png'), dpi=300)
    plt.close()
    
    # Save the multi-view visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Find the best slices for visualization
    tumor_slice_sizes_z = [np.sum(tumor_mask[:, :, i]) for i in range(tumor_mask.shape[2])]
    tumor_slice_sizes_y = [np.sum(tumor_mask[:, i, :]) for i in range(tumor_mask.shape[1])]
    tumor_slice_sizes_x = [np.sum(tumor_mask[i, :, :]) for i in range(tumor_mask.shape[0])]
    
    mid_z = mri_data.shape[2] // 2
    mid_y = mri_data.shape[1] // 2
    mid_x = mri_data.shape[0] // 2
    
    best_z = np.argmax(tumor_slice_sizes_z) if max(tumor_slice_sizes_z) > 0 else mid_z
    best_y = np.argmax(tumor_slice_sizes_y) if max(tumor_slice_sizes_y) > 0 else mid_y
    best_x = np.argmax(tumor_slice_sizes_x) if max(tumor_slice_sizes_x) > 0 else mid_x
    
    # Axial view (z-axis)
    axes[0, 0].imshow(mri_data[:, :, best_z], cmap='gray')
    axes[0, 0].set_title(f'Axial (z={best_z})')
    axes[0, 0].axis('off')
    
    # Axial view with tumor overlay
    axes[1, 0].imshow(mri_data[:, :, best_z], cmap='gray')
    mask_z = np.zeros((*mri_data[:, :, best_z].shape, 4))
    mask_z[:, :, 0] = 1.0  # Red channel
    mask_z[:, :, 3] = tumor_mask[:, :, best_z] * 0.7  # Alpha channel
    axes[1, 0].imshow(mask_z)
    axes[1, 0].set_title('Tumor Overlay (Axial)')
    axes[1, 0].axis('off')
    
    # Coronal view (y-axis)
    axes[0, 1].imshow(mri_data[:, best_y, :].T, cmap='gray')
    axes[0, 1].set_title(f'Coronal (y={best_y})')
    axes[0, 1].axis('off')
    
    # Coronal view with tumor overlay
    axes[1, 1].imshow(mri_data[:, best_y, :].T, cmap='gray')
    mask_y = np.zeros((*mri_data[:, best_y, :].T.shape, 4))
    mask_y[:, :, 0] = 1.0  # Red channel
    mask_y[:, :, 3] = tumor_mask[:, best_y, :].T * 0.7  # Alpha channel
    axes[1, 1].imshow(mask_y)
    axes[1, 1].set_title('Tumor Overlay (Coronal)')
    axes[1, 1].axis('off')
    
    # Sagittal view (x-axis)
    axes[0, 2].imshow(mri_data[best_x, :, :].T, cmap='gray')
    axes[0, 2].set_title(f'Sagittal (x={best_x})')
    axes[0, 2].axis('off')
    
    # Sagittal view with tumor overlay
    axes[1, 2].imshow(mri_data[best_x, :, :].T, cmap='gray')
    mask_x = np.zeros((*mri_data[best_x, :, :].T.shape, 4))
    mask_x[:, :, 0] = 1.0  # Red channel
    mask_x[:, :, 3] = tumor_mask[best_x, :, :].T * 0.7  # Alpha channel
    axes[1, 2].imshow(mask_x)
    axes[1, 2].set_title('Tumor Overlay (Sagittal)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tumor_multiview.png'), dpi=300)
    plt.close()
    
    # Save tumor mask as a NumPy array
    np.save(os.path.join(output_dir, 'tumor_mask.npy'), tumor_mask)
    
    print(f"Results saved to {output_dir}")

def analyze_mri(mri_file_path, mask_file_path=None, threshold_factor=2.0):
    print("Loading MRI data...")
    mri_data, affine = load_mri_data(mri_file_path)
    
    print(f"MRI dimensions: {mri_data.shape}")
    print(f"Value range: {np.min(mri_data)} to {np.max(mri_data)}")
    
    # Calculate voxel dimensions from the affine matrix
    voxel_dimensions = np.abs(np.diagonal(affine)[:3])
    
    # Either load tumor mask or perform segmentation
    if mask_file_path and os.path.exists(mask_file_path):
        print(f"Loading tumor mask from: {mask_file_path}")
        tumor_mask, mask_affine = load_tumor_mask(mask_file_path)
        
        # Check if the mask and MRI have the same dimensions
        if tumor_mask.shape != mri_data.shape:
            print(f"Warning: Tumor mask dimensions {tumor_mask.shape} don't match MRI dimensions {mri_data.shape}")
            print("Attempting to resample mask to match MRI...")
            
            from scipy.ndimage import zoom
            zoom_factors = tuple(float(m) / float(t) for m, t in zip(mri_data.shape, tumor_mask.shape))
            tumor_mask = zoom(tumor_mask, zoom_factors, order=0)  # order=0 for nearest-neighbor interpolation
            print(f"Resampled mask dimensions: {tumor_mask.shape}")
    else:
        print("No tumor mask provided, performing automatic segmentation...")
        tumor_mask = segment_tumor(mri_data, threshold_factor)
    
    # Calculate tumor statistics
    print("Calculating tumor statistics...")
    tumor_stats = calculate_tumor_statistics(tumor_mask, voxel_dimensions)
    
    print(f"Tumor volume: {tumor_stats['volume_cm3']:.2f} cm³ ({tumor_stats['voxel_count']} voxels)")
    print(f"Tumor dimensions (mm): {tumor_stats['dimensions_mm'][0]:.2f} × {tumor_stats['dimensions_mm'][1]:.2f} × {tumor_stats['dimensions_mm'][2]:.2f}")
    
    
    # Display multiple slices with tumor
    print("Displaying axial slices with tumor...")
    display_mri_slices(mri_data, tumor_mask, num_slices=5)
    
    return mri_data, tumor_mask, tumor_stats

def main():
    mri_file_path = r"C:\Users\erick\Documents\Python Scripts\MRI_Tumor_Visualization\PatientID_0003_Timepoint_1_brain_t1c.nii"
    mask_file_path = r"C:\Users\erick\Documents\Python Scripts\MRI_Tumor_Visualization\PatientID_0003_Timepoint_1_tumorMask.nii"
    
    # Check if the MRI file exists
    if os.path.exists(mri_file_path):
        print(f"Processing MRI file: {mri_file_path}")
        

        print("\n--- RUNNING AUTOMATIC SEGMENTATION ---")
        print("Using automatic segmentation.")
        threshold_factor = 2.5 # Higher values = more selective segmentation
        auto_mri_data, auto_tumor_mask, auto_tumor_stats = analyze_mri(mri_file_path, None, threshold_factor)
        
        auto_output_dir = os.path.join(os.path.dirname(mri_file_path), "auto_tumor_analysis_results")
        save_results(auto_mri_data, auto_tumor_mask, auto_output_dir, auto_tumor_stats)

        print("\n--- RUNNING MASK-BASED SEGMENTATION ---")
        if os.path.exists(mask_file_path):
            print(f"Using provided tumor mask: {mask_file_path}")
            mask_mri_data, mask_tumor_mask, mask_tumor_stats = analyze_mri(mri_file_path, mask_file_path)
            
            mask_output_dir = os.path.join(os.path.dirname(mri_file_path), "mask_tumor_analysis_results")
            save_results(mask_mri_data, mask_tumor_mask, mask_output_dir, mask_tumor_stats)
        else:
            print(f"Error: Mask file not found at {mask_file_path}")
    else:
        print(f"Error: MRI file not found at {mri_file_path}")

main()
