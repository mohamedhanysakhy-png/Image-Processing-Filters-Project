# Real-Time Image Processing & Frequency Domain Toolbox

This project is a high-performance, interactive Image procassing application built using **Python**, **OpenCV**, and **NumPy**. It implements core Digital Image Processing (DIP) concepts—ranging from spatial filtering to frequency domain analysis—applied directly to a live webcam stream.

---

##  Project Overview

This application serves as a comprehensive "sandbox" for image processing. It allows users to toggle between various filtering techniques in real-time, providing immediate visual feedback on how mathematical kernels and transformations affect digital signals (images).

### Core Components:
1.  **Manual Canny Implementation**: A step-by-step implementation of the Canny algorithm.
2.  **Spatial Domain Restoration**: Various mean and order-statistic filters.
3.  **Frequency Domain Filtering**: Implementation of DFT with Ideal, Gaussian, and Butterworth filters.

---

##  Technical Detailed Breakdown

### 1. Edge Detection (Spatial Domain)
The project features a **manual implementation** of the Canny Edge Detection algorithm, which follows these specific stages:
* **Noise Reduction**: Application of a Gaussian Blur.
* **Gradient Calculation**: Using Sobel kernels to find intensity gradients ($G_x$ and $G_y$).
* **Non-Maximum Suppression (NMS)**: Thinning out the edges by suppressing pixels that are not local maxima in the direction of the gradient.
* **Double Thresholding**: Classifying pixels as "Strong," "Weak," or "Non-edges."
* **Hysteresis Tracking**: Connecting weak edges to strong edges to finalize the edge map.



### 2. Image Restoration Filters
We implement two main categories of spatial filters to handle different types of image noise:
* **Mean Filters**: Arithmetic, Geometric, Harmonic, and Contraharmonic means (useful for Gaussian and Salt-and-Pepper noise).
* **Order-Statistics Filters**: Median, Min, Max, Midpoint, and Alpha-trimmed mean filters (highly effective for impulse noise).



### 3. Frequency Domain Processing
The application utilizes the **Discrete Fourier Transform (DFT)** to transition the image from the spatial domain to the frequency domain.
* **Transform**: `cv2.dft` is used to compute the complex output.
* **Shifting**: The zero-frequency component is shifted to the center of the spectrum for easier filtering.
* **Masking**: Low-Pass (LPF), High-Pass (HPF), Band-Pass (BPF), and Band-Reject (BRF) masks are applied.
* **Inverse Transform**: The filtered signal is converted back to the spatial domain using `cv2.idft`.



---

##  Interactive Controls

Run the script and use these keys to interact with the live feed:

| Category | Keys | Function |
| :--- | :--- | :--- |
| **Basic** | `o` / `g` | Original Color / Grayscale |
| **Edges** | `c` / `s` / `l` | Canny / Sobel / Laplacian |
| **Mean** | `e` / `r` / `h` / `v` | Arithmetic / Geometric / Harmonic / Contraharmonic |
| **Order Stats** | `M` / `,` / `.` / `/` | Median / Min / Max / Midpoint |
| **Freq (Low)** | `1` / `2` / `3` | Ideal / Gaussian / Butterworth LPF |
| **Freq (High)** | `4` / `5` / `6` | Ideal / Gaussian / Butterworth HPF |
| **Tuning** | `p` / `m` | Increase / Decrease Kernel Size |
| **Tuning** | `z` / `a` | Adjust Cutoff Frequency ($D_0$) |

---

##  Requirements

* **Python 3.x**
* **OpenCV** (`pip install opencv-python`)
* **NumPy** (`pip install numpy`)

---
