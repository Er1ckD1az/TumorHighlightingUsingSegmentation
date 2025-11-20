# 🧠 MRI Tumor Segmentation & Visualization

An automated medical imaging analysis tool that uses computer vision techniques to detect, segment, and visualize brain tumors from MRI scans. The system processes NIfTI format brain MRI data and generates tumor analysis reports with multi-view visualizations. This project was developed as part of my exploration into medical imaging and neuroscience, as instead of using simple computer vision techniques, I planned to use a custom built segmentation model. You can find the culmination of said work in my other project: (see [Metis DICOM viewer?](https://github.com/Er1ckD1az/Metis_DICOM_Viewer))

## 🔬 How Does It Work?

This project combines image processing algorithms and morphological operations to identify tumor regions in brain MRI scans(no segmentation models this go around). It supports both automatic segmentation using adaptive thresholding and manual segmentation using pre-labeled tumor masks.

## 🛠️ System Architecture

The system can be broken down into **five main components**:

### 1️⃣ MRI Data Loading & Preprocessing
- **Purpose:** Import and prepare NIfTI format MRI scans for analysis
- **Process:**
  - Loads 3D MRI volumes using the NiBabel library
  - Extracts affine transformation matrices for spatial calibration
  - Calculates voxel dimensions for accurate volume measurements

### 2️⃣ Tumor Segmentation
The project has two segmentation approaches:

#### Automatic Segmentation
- **Technique:** Otsu's thresholding with adaptive adjustment
- **Process:**
  1. Normalizes MRI intensity values to 0-1 range
  2. Applies Otsu's method to determine optimal threshold
  3. Adjusts threshold by a configurable factor (default: 2.0-2.5x)
  4. Creates binary mask of potential tumor regions
- **Refinement:** Morphological operations clean up the mask:
  - Removes small isolated objects (< 100 voxels)
  - Applies binary closing with spherical structuring element
  - Identifies largest connected component as primary tumor

#### Manual Segmentation
- **Purpose:** Uses expert-labeled tumor masks for validation
- **Features:**
  - Loads pre-existing tumor masks in NIfTI format
  - Automatically resamples masks to match MRI dimensions
  - Validates mask-MRI alignment

### 3️⃣ Statistical Analysis
- **Tumor Volume Calculation:**
  - Voxel counting with precise spatial calibration
  - Converts to both cubic millimeters and cubic centimeters
- **Dimensional Analysis:**
  - Calculates tumor extent in all three axes (X, Y, Z)
  - Determines bounding box coordinates
  - Computes center of mass for tumor localization
- **Output Metrics:**
  - Total voxel count
  - Volume in mm³ and cm³
  - Physical dimensions in millimeters
  - Spatial bounds in voxel coordinates
  - Center of mass coordinates

### 4️⃣ Multi-View Visualization
The system generates comprehensive visualizations across three anatomical planes:

#### Axial View (Top-Down)
- Shows horizontal slices through the brain
- Ideal for viewing tumor cross-sections
- Automatically selects slice with largest tumor area

#### Coronal View (Front-Back)
- Displays frontal brain sections
- Useful for understanding anterior-posterior tumor extent
- Shows tumor position relative to brain structures

#### Sagittal View (Side)
- Presents lateral brain slices
- Reveals tumor depth and vertical extent
- Helpful for surgical planning perspectives

**Visualization Features:**
- Side-by-side comparison: original MRI vs. tumor overlay
- Red semi-transparent overlay (70% opacity) for clear tumor visualization
- Intelligent slice selection prioritizing maximum tumor visibility
- High-resolution output (300 DPI) for clinical review

### 5️⃣ Results Export & Documentation
- **Statistical Report:** Text file with complete tumor metrics
- **2D Visualization:** Best single-slice view with overlay
- **3D Multi-View:** Six-panel comprehensive visualization
- **Tumor Mask:** NumPy array for further analysis
- **Organized Output:** Separate directories for automatic vs. manual segmentation results

## :chart_with_downwards_trend: Exploring Output
Now that we've discussed all the techincal jargin of the project. It's time to see how it actaully preformed. And As you can see from the image below. Not so great.


![Image showing a comparison of tumor detection at different threshold values](./MRI_Tumor_Visualization/Tumor_Comparison_analysis.png)

| Threshold Factor | Tumor Volume (cm^3) | Voxel Count | Relative error |
| :-------------: | :-------------: | :-------------: | :-------------: |
| Ground Truth | 84.54 | 84,540 | ------ |
| 2.0 | 950.97 | 950,966 | +1025% overestimation |
| 2.5 | 55.22 | 55,221 | -34.7% underestimation |
| 3.0 | 10.13 | 10,135 | -88.0% understimation |

### Analysis

Why did it preform so poorly? The fundamental method imaging technique we used is called Otsu's method. However, this comes with its own drawbacks. Tumor tissue and healthy brain tissue often have overlapping intensity distributions in MRI scans. Otsu's algorithm assumes dimodal intensity distributions (2 distinct peaks), but brain MRI data is inherently complex. Some other issues include:
- **No Spatial context:** Thresholding operates on pixel by pixel basis without considering spatial relationships.
- **Single Global Threshold Limitation** Using one threshold value for the entire 3D volume is overly simplistic.
- **Sensitivity to Threshold Factor:** The results table demonstrates extreme sensitivity to the threshold multiplier. 

So, how could we improve the output? As stated earlier, the best approach would be the usuage of Deep Learning models. Models such as U-Net and PSP-Net would drastically improve the accuracy of the highlighting. 

Overall, while the output did not produce very accuracte results. I achieved what I set out to do with the project. Which was to serve as an introduction to neuroimaging, so that I could take the lessons learned and continue my work in my capstone project. 

## 📚 Lessons Learned

This project sparked my interest for medical imaging and nuroscience. So much, that I hope to start a carerr in the medical field and went on to heavily influence my future projects.

**Technical Growth:**
- **Medical Imaging Fundamentals:** Understanding NIfTI formats, spatial transformations, and anatomical coordinate systems
- **Computer Vision Techniques:** Hands-on experience with segmentation, thresholding, morphological operations, and connected component analysis
- **Algorithm Optimization:** Balancing sensitivity vs. specificity in tumor detection, learning when automatic methods excel and when expert validation is necessary

**Domain Knowledge:**
- **Brain Anatomy:** Learning to interpret axial, coronal, and sagittal views and understanding spatial relationships in neuroimaging
- **Clinical Applications:** Recognizing how volume measurements and tumor localization inform treatment decisions

