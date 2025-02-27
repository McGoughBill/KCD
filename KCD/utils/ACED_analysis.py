import os
import numpy as np
import nibabel as nib
import pandas as pd
from skimage import measure

def get_axial_diameter(data, spacing):
    """
    Calculate the maximum axial diameter of the largest connected region from a 3D binary segmentation.

    The axial diameter is defined as the maximum major axis length observed across all axial slices
    (slices along the last dimension) of the largest connected region, converted into physical units
    using the in-plane voxel spacing.

    Parameters:
        data (ndarray): A 3D binary numpy array representing the segmentation.
        spacing (tuple or list): The voxel spacing in (row, column, axial) directions.
                                 For in-plane calculations, spacing[0] and spacing[1] are used.

    Returns:
        float: The maximum axial diameter in physical units. Returns 0 if no valid region is found.
    """
    # Label connected regions in the 3D volume.
    labelled = measure.label(data)
    props = measure.regionprops(labelled)
    if not props:
        return 0.0  # No regions found

    # Isolate the largest connected component.
    largest_region = max(props, key=lambda x: x.area)
    largest_mask = labelled == largest_region.label

    axial_diameter_max = 0.0

    # Loop through each axial slice.
    for i in range(largest_mask.shape[-1]):
        slice_data = largest_mask[:, :, i]
        if np.sum(slice_data) == 0:
            continue

        slice_label = measure.label(slice_data)
        slice_props = measure.regionprops(slice_label)
        if not slice_props:
            continue

        # Select the largest region in the slice.
        slice_largest = max(slice_props, key=lambda x: x.area)
        L = slice_largest.major_axis_length
        theta = slice_largest.orientation  # Angle (in radians) of the major axis with respect to the horizontal

        # Convert the pixel-based length to physical units using in-plane spacing.
        physical_L = L * np.sqrt((np.cos(theta) * spacing[1])**2 + (np.sin(theta) * spacing[0])**2)
        axial_diameter_max = max(axial_diameter_max, physical_L)

    return axial_diameter_max

def get_sphericity(data, spacing):
    """
    Calculate the sphericity of the largest connected region in a 3D binary volume.

    Sphericity is defined as the ratio of the surface area of a sphere (with the same volume as the region)
    to the actual surface area of the region. A perfect sphere has a sphericity of 1.

    Parameters:
        data (ndarray): A 3D binary numpy array representing the segmentation.
        spacing (tuple or list): The voxel spacing along each dimension (dx, dy, dz).

    Returns:
        float: The sphericity of the segmented region. Returns None if no region is found.
    """
    labelled = measure.label(data)
    props = measure.regionprops(labelled)
    if not props:
        return None

    # Extract the largest connected component.
    largest_region = max(props, key=lambda x: x.area)
    mask = labelled == largest_region.label

    # Compute volume (number of voxels times the voxel volume).
    volume = np.sum(mask) * np.prod(spacing)

    # Compute the surface area using marching cubes.
    verts, faces, normals, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
    surface_area = measure.mesh_surface_area(verts, faces)

    # Compute the expected surface area of a sphere with the same volume.
    expected_surface_area = 4 * np.pi * (volume / ((4/3) * np.pi)) ** (2/3)

    return expected_surface_area / surface_area

def parse_filename(filename):
    """
    Parse the filename to extract metadata such as artist ID, scan ID, and scan date.

    Expected filename format:
        ARTISTXXX_AAA-YYYYMMDD-BBB_<string>.nii.gz

    Returns:
        dict: Dictionary with keys 'artist_id', 'scan_id', 'year', 'month', and 'day'.
              If parsing fails, returns an empty dictionary.
    """
    try:
        base = os.path.basename(filename)
        parts = base.split('_')
        artist_id = parts[0][6:]
        scan_info = parts[1].split('-')
        scan_id = scan_info[0]
        date_str = scan_info[1]
        year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
        return {'artist_id': artist_id, 'scan_id': scan_id, 'year': year, 'month': month, 'day': day}
    except Exception as e:
        print(f"Filename parsing error for {filename}: {e}")
        return {}

def analyze_segmentation(filepath):
    """
    Load a Nifti segmentation file, convert it to a binary mask,
    and compute morphological properties for the largest connected region.

    Parameters:
        filepath (str): Path to the Nifti file.

    Returns:
        dict: Dictionary containing 'volume', 'sphericity', and 'axial_diameter' in physical units.
              Returns None if no valid region is found.
    """
    try:
        img = nib.load(filepath)
        spacing = img.header.get_zooms()
        data = np.round(img.get_fdata()).astype(int)
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

    labelled = measure.label(data)
    props = measure.regionprops(labelled)
    if not props:
        return None

    largest_region = max(props, key=lambda x: x.area)
    volume = largest_region.area * np.prod(spacing)
    sphericity = get_sphericity(data, spacing)
    axial_diameter = get_axial_diameter(data, spacing)

    return {'volume': volume, 'sphericity': sphericity, 'axial_diameter': axial_diameter}

def main(meta_folder: str):
    """
    Perform end-to-end morphology analysis on a set of Nifti segmentation files.

    The analysis is carried out on two sets of segmentations produced with different thresholds
    (e.g., 'high_acc' and 'high_spec'). For each scan with a positive detection in the high_acc directory,
    the corresponding high_spec scan is also analyzed if available. For each scan, the volume, sphericity,
    and axial diameter of the largest connected region are computed. Additionally, the axial diameters from
    both segmentation types are compared, flagging similarity if they are within 10%.

    Results are saved as a CSV file in the meta_folder.
    """
    high_acc_dir = os.path.join(meta_folder, 'high_acc', 'positives_nii')
    high_spec_dir = os.path.join(meta_folder, 'high_spec', 'positives_nii')

    # Get all high_acc segmentation files.
    high_acc_files = [f for f in os.listdir(high_acc_dir) if f.endswith('.nii.gz')]

    results = []
    for filename in high_acc_files:
        print(f"Processing file: {filename}")
        entry = parse_filename(filename)

        ha_fp = os.path.join(high_acc_dir, filename)
        hs_fp = os.path.join(high_spec_dir, filename)

        # Analyze high_acc segmentation.
        ha_props = analyze_segmentation(ha_fp)
        if ha_props is None:
            print(f"No valid region found in high_acc file: {filename}")
            continue

        entry['ha_volume'] = ha_props['volume']
        entry['ha_axial_diameter'] = ha_props['axial_diameter']
        entry['ha_sphericity'] = ha_props['sphericity']

        # Analyze high_spec segmentation if the file exists.
        if os.path.exists(hs_fp):
            hs_props = analyze_segmentation(hs_fp)
            if hs_props is not None:
                entry['hs_volume'] = hs_props['volume']
                entry['hs_axial_diameter'] = hs_props['axial_diameter']
                entry['hs_sphericity'] = hs_props['sphericity']

                # Flag similarity if the axial diameters are within 10%.
                if ha_props['axial_diameter'] > 0:
                    entry['diam_similar'] = 1 if abs(ha_props['axial_diameter'] - hs_props['axial_diameter']) / ha_props['axial_diameter'] < 0.1 else 0
                else:
                    entry['diam_similar'] = 0
            else:
                entry['hs_volume'] = 0
                entry['hs_axial_diameter'] = 0
                entry['hs_sphericity'] = 0
                entry['diam_similar'] = 0
        else:
            entry['hs_volume'] = 0
            entry['hs_axial_diameter'] = 0
            entry['hs_sphericity'] = 0
            entry['diam_similar'] = 0

        results.append(entry)
        print(entry)
        print()

    # Save the results to a CSV file.
    df = pd.DataFrame(results)
    output_csv = os.path.join(meta_folder, 'ACED_analysis.csv')
    df.to_csv(output_csv, index=False)
    print(f"Analysis complete. Results saved to {output_csv}")

if __name__ == '__main__':
    meta_folder = '/Users/mcgoug01/Downloads/ACED_detections'
    main(meta_folder)
