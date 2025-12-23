import cv2
import numpy as np

# -------------------- Helpers --------------------
def make_odd(k):
    return k if k % 2 == 1 else k + 1

def gaussian_blur(img, ksize=3, sigma=1):
    k = make_odd(max(3, ksize))
    return cv2.GaussianBlur(img, (k, k), sigma)

def compute_gradients(img, ksize=3):
    # (Used by my_canny) keep it because NMS needs direction
    img = img.astype(np.float64)
    k = make_odd(max(3, ksize))
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)

    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    mmax = grad_mag.max()
    grad_mag = (grad_mag / mmax * 255) if mmax > 1e-6 else np.zeros_like(grad_mag)

    grad_dir = np.arctan2(grad_y, grad_x)
    return grad_mag, grad_dir

def sobel_x(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gx = np.absolute(gx)
    m = gx.max()
    gx = (gx / m * 255) if m > 1e-6 else np.zeros_like(img_gray, dtype=np.float64)
    return gx.astype(np.uint8)

def sobel_y(img_gray):
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gy = np.absolute(gy)
    m = gy.max()
    gy = (gy / m * 255) if m > 1e-6 else np.zeros_like(img_gray, dtype=np.float64)
    return gy.astype(np.uint8)

def sobel_edges(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    m = mag.max()
    mag = (mag / m * 255) if m > 1e-6 else np.zeros_like(img_gray, dtype=np.float64)
    return mag.astype(np.uint8)

# -------------------- Canny (your implementation) --------------------
def non_maximum_suppression(grad_mag, grad_dir):
    M, N = grad_mag.shape
    Z = np.zeros((M, N), dtype=np.float64)
    angle = grad_dir * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q, r = 0, 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = grad_mag[i, j + 1]; r = grad_mag[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = grad_mag[i + 1, j - 1]; r = grad_mag[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = grad_mag[i + 1, j]; r = grad_mag[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = grad_mag[i - 1, j - 1]; r = grad_mag[i + 1, j + 1]

            Z[i, j] = grad_mag[i, j] if (grad_mag[i, j] >= q and grad_mag[i, j] >= r) else 0

    return Z

def double_threshold(img, low_thresh, high_thresh):
    strong, weak = 255, 75
    res = np.zeros_like(img, dtype=np.uint8)

    strong_i, strong_j = np.where(img >= high_thresh)
    weak_i, weak_j = np.where((img >= low_thresh) & (img < high_thresh))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

def hysteresis(img, weak=75, strong=255):
    M, N = img.shape
    res = img.copy()
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if res[i, j] == weak:
                if np.any(res[i - 1:i + 2, j - 1:j + 2] == strong):
                    res[i, j] = strong
                else:
                    res[i, j] = 0
    return res

def my_canny(img_gray, low_thresh=20, high_thresh=60, ksize=3, sigma=1):
    blurred = gaussian_blur(img_gray, ksize, sigma)
    mag, direction = compute_gradients(blurred, ksize=ksize)
    nms = non_maximum_suppression(mag, direction)
    dt, weak, strong = double_threshold(nms, low_thresh, high_thresh)
    edges = hysteresis(dt, weak=weak, strong=strong)
    return edges.astype(np.uint8)

def laplacian_filter2d(img_gray):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    lap = cv2.filter2D(img_gray, cv2.CV_64F, kernel)
    lap = np.absolute(lap)
    lap = np.clip(lap, 0, 255)
    return lap.astype(np.uint8)

# -------------------- Mean Filters (spatial restoration) --------------------
def arithmetic_mean_filter(img_gray, ksize=3):
    k = make_odd(max(3, ksize))
    return cv2.blur(img_gray, (k, k))

def geometric_mean_filter(img_gray, ksize=3):
    k = make_odd(max(3, ksize))
    eps = 1e-6
    g = img_gray.astype(np.float32)
    logg = np.log(g + eps)
    m = cv2.blur(logg, (k, k))
    out = np.exp(m)
    return np.clip(out, 0, 255).astype(np.uint8)

def harmonic_mean_filter(img_gray, ksize=3):
    k = make_odd(max(3, ksize))
    eps = 1e-6
    g = img_gray.astype(np.float32)
    inv = 1.0 / (g + eps)
    s = cv2.blur(inv, (k, k))  
    mn = float(k * k)
    out = mn / (s * mn + eps)
    return np.clip(out, 0, 255).astype(np.uint8)

def contraharmonic_mean_filter(img_gray, ksize=3, q=1.5):
    k = make_odd(max(3, ksize))
    eps = 1e-6
    g = img_gray.astype(np.float32)

    gq = np.power(g + eps, q)
    gq1 = np.power(g + eps, q + 1.0)

    mean_gq = cv2.blur(gq, (k, k))
    mean_gq1 = cv2.blur(gq1, (k, k))

    out = mean_gq1 / (mean_gq + eps)
    return np.clip(out, 0, 255).astype(np.uint8)

# -------------------- Order Statistics Filters --------------------
def median_filter(img_gray, ksize=3):
    k = make_odd(max(3, ksize))
    return cv2.medianBlur(img_gray, k)

def min_filter(img_gray, ksize=3):
    k = make_odd(max(3, ksize))
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(img_gray, kernel, iterations=1)

def max_filter(img_gray, ksize=3):
    k = make_odd(max(3, ksize))
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(img_gray, kernel, iterations=1)

def midpoint_filter(img_gray, ksize=3):
    mn = min_filter(img_gray, ksize=ksize).astype(np.float32)
    mx = max_filter(img_gray, ksize=ksize).astype(np.float32)
    out = (mn + mx) / 2.0
    return np.clip(out, 0, 255).astype(np.uint8)

def alpha_trimmed_mean_filter(img_gray, ksize=3, d=2):
    k = make_odd(max(3, ksize))
    n = k * k
    d = int(max(0, d))
    if d % 2 == 1:
        d += 1
    if d >= n:
        d = n - 1
        if d % 2 == 1:
            d -= 1

    pad = k // 2
    g = img_gray.astype(np.float32)
    gp = np.pad(g, ((pad, pad), (pad, pad)), mode='edge')

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(gp, (k, k))
        win = win.reshape(win.shape[0], win.shape[1], n)
        win.sort(axis=2)
        lo = d // 2
        hi = n - (d // 2)
        trimmed = win[:, :, lo:hi]
        out = trimmed.mean(axis=2)
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception:
        h, w = img_gray.shape
        out = np.zeros((h, w), np.float32)
        lo = d // 2
        hi = n - (d // 2)
        for i in range(h):
            for j in range(w):
                block = gp[i:i + k, j:j + k].reshape(-1)
                block.sort()
                out[i, j] = block[lo:hi].mean()
        return np.clip(out, 0, 255).astype(np.uint8)

# -------------------- Frequency Domain Filtering --------------------
def _distance_grid(rows, cols):
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    D = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    return D

def _lp_mask(rows, cols, filter_name, D0, n=2, W=10):
    D = _distance_grid(rows, cols).astype(np.float32)
    D0 = max(1.0, float(D0))
    n = max(1, int(n))
    W = max(1.0, float(W))

    if filter_name == 'ILPF':
        H = (D <= D0).astype(np.float32)
    elif filter_name == 'GLPF':
        H = np.exp(-(D ** 2) / (2 * (D0 ** 2))).astype(np.float32)
    elif filter_name == 'BLPF':
        H = (1.0 / (1.0 + (D / D0) ** (2 * n))).astype(np.float32)

    elif filter_name == 'IHPF':
        H = (D > D0).astype(np.float32)
    elif filter_name == 'GHPF':
        H = (1.0 - np.exp(-(D ** 2) / (2 * (D0 ** 2)))).astype(np.float32)
    elif filter_name == 'BHPF':
        H = (1.0 - (1.0 / (1.0 + (D / D0) ** (2 * n)))).astype(np.float32)

    else:
        if filter_name == 'IBRF':
            H = np.ones((rows, cols), np.float32)
            H[np.abs(D - D0) <= (W / 2.0)] = 0.0
        elif filter_name == 'IBPF':
            H = np.zeros((rows, cols), np.float32)
            H[np.abs(D - D0) <= (W / 2.0)] = 1.0

        elif filter_name == 'GBRF':
            eps = 1e-6
            term = ((D ** 2 - D0 ** 2) / (np.maximum(D, eps) * W)) ** 2
            H = (1.0 - np.exp(-term)).astype(np.float32)
        elif filter_name == 'GBPF':
            eps = 1e-6
            term = ((D ** 2 - D0 ** 2) / (np.maximum(D, eps) * W)) ** 2
            H = (np.exp(-term)).astype(np.float32)

        elif filter_name == 'BBRF':
            eps = 1e-6
            denom = 1.0 + ((D * W) / (np.maximum(np.abs(D ** 2 - D0 ** 2), eps))) ** (2 * n)
            H = (1.0 / denom).astype(np.float32)
        elif filter_name == 'BBPF':
            eps = 1e-6
            denom = 1.0 + ((D * W) / (np.maximum(np.abs(D ** 2 - D0 ** 2), eps))) ** (2 * n)
            H = (1.0 - (1.0 / denom)).astype(np.float32)
        else:
            H = np.ones((rows, cols), np.float32)

    return H

def apply_freq_filter_gray(img_gray, filter_name='ILPF', D0=30, n=2, W=10):
    img = img_gray.astype(np.float32)
    rows, cols = img.shape

    dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    H = _lp_mask(rows, cols, filter_name, D0=D0, n=n, W=W)
    H2 = np.repeat(H[:, :, np.newaxis], 2, axis=2)

    fshift = dft_shift * H2

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    out = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mask_vis = (H * 255).astype(np.uint8)

    mag = cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])
    spec = np.log(mag + 1.0)
    spec = cv2.normalize(spec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return out, mask_vis, spec

def put_hud(img, text_lines):
    if img.ndim == 2:
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img.copy()

    y = 25
    for line in text_lines:
        cv2.putText(canvas, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 0, 0), 2, cv2.LINE_AA)
        y += 25
    return canvas

def stack3(a, b, c):
    def to3(x):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if x.ndim == 2 else x
    A, B, C = to3(a), to3(b), to3(c)
    h = min(A.shape[0], B.shape[0], C.shape[0])
    A = cv2.resize(A, (A.shape[1], h))
    B = cv2.resize(B, (B.shape[1], h))
    C = cv2.resize(C, (C.shape[1], h))
    return np.hstack([A, B, C])

# -------------------- Main Project --------------------
def webcam_edge_freq_view():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    mode = 'g'
    ksize = 3
    sigma = 1
    low_t = 20
    high_t = 60

    contra_q = 1.5
    alpha_d = 2

    D0 = 30
    order_n = 2
    band_W = 20
    freq_filter = 'ILPF'

    print("\n==== MODES ====")
    print("o = original")
    print("g = gray")
    print("b = blur only")
    print("c = my canny")
    print("s = sobel magnitude (your sobel_edges)")
    print("x = sobel x (your sobel_x)")
    print("y = sobel y (your sobel_y)")
    print("l = laplacian (0 1 0 / 1 -4  1 / 0 1 0)")

    print("\n---- mean filters ----")
    print("e = arithmetic mean")
    print("r = geometric mean")
    print("h = harmonic mean")
    print("v = contraharmonic mean")

    print("\n---- order statistics filters ----")
    print("M = median")
    print(", = min")
    print(". = max")
    print("/ = midpoint")
    print("; = alpha trimmed mean")

    print("\n==== FREQ FILTER MODES (press these, mode becomes 'f') ====")
    print("1=ILPF  2=GLPF  3=BLPF")
    print("4=IHPF  5=GHPF  6=BHPF")
    print("7=IBRF  8=GBRF  9=BBRF")
    print("0=IBPF  -=GBPF  ==BBPF")

    print("\n==== CONTROLS ====")
    print("p/m : increase/decrease ksize (spatial)")
    print("i/k : increase/decrease high threshold (canny)")
    print("j/n : increase/decrease low threshold  (canny)")
    print("z/a : D0 + / -   (frequency cutoff)")
    print("u/d : band W + / - (for band filters)")
    print("r/f : order n + / - (butterworth)")
    print("[/] : contraharmonic q - / +")
    print("\\/' : alpha d - / +   (even, for alpha trimmed)")
    print("q   : quit\n")

    window_name = "Edge + Frequency Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if mode == 'o':
            show = frame

        elif mode == 'g':
            show = gray

        elif mode == 'b':
            show = gaussian_blur(gray, ksize=ksize, sigma=sigma)

        elif mode == 'c':
            show = my_canny(gray, low_thresh=low_t, high_thresh=high_t, ksize=ksize, sigma=sigma)

        elif mode == 's':
            show = sobel_edges(gray)

        elif mode == 'x':
            show = sobel_x(gray)

        elif mode == 'y':
            show = sobel_y(gray)

        elif mode == 'l':
            show = laplacian_filter2d(gray)

        # mean filters
        elif mode == 'e':
            show = arithmetic_mean_filter(gray, ksize=ksize)
        elif mode == 'r':
            show = geometric_mean_filter(gray, ksize=ksize)
        elif mode == 'h':
            show = harmonic_mean_filter(gray, ksize=ksize)
        elif mode == 'v':
            show = contraharmonic_mean_filter(gray, ksize=ksize, q=contra_q)

        # order statistics filters
        elif mode == 'M':
            show = median_filter(gray, ksize=ksize)
        elif mode == ',':
            show = min_filter(gray, ksize=ksize)
        elif mode == '.':
            show = max_filter(gray, ksize=ksize)
        elif mode == '/':
            show = midpoint_filter(gray, ksize=ksize)
        elif mode == ';':
            show = alpha_trimmed_mean_filter(gray, ksize=ksize, d=alpha_d)

        elif mode == 'f':
            out, mask_vis, spec_vis = apply_freq_filter_gray(
                gray, filter_name=freq_filter, D0=D0, n=order_n, W=band_W
            )
            show = stack3(out, mask_vis, spec_vis)

        else:
            show = frame

        hud_lines = [
            f"mode:{mode} | ksize:{ksize} | sigma:{sigma} | canny low:{low_t} high:{high_t}",
            f"freq:{freq_filter} | D0:{D0} | n:{order_n} | W:{band_W}",
            f"contra q:{contra_q:.2f} | alpha d:{alpha_d}",
        ]
        show_hud = put_hud(show, hud_lines)

        cv2.imshow(window_name, show_hud)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # modes
        elif key == ord('o'):
            mode = 'o'
        elif key == ord('g'):
            mode = 'g'
        elif key == ord('b'):
            mode = 'b'
        elif key == ord('c'):
            mode = 'c'
        elif key == ord('s'):
            mode = 's'
        elif key == ord('x'):
            mode = 'x'
        elif key == ord('y'):
            mode = 'y'
        elif key == ord('l'):
            mode = 'l'

        # mean filter modes
        elif key == ord('e'):
            mode = 'e'
        elif key == ord('r'):
            mode = 'r'
        elif key == ord('h'):
            mode = 'h'
        elif key == ord('v'):
            mode = 'v'

        # order statistics modes
        elif key == ord('M'):
            mode = 'M'
        elif key == ord(','):
            mode = ','
        elif key == ord('.'):
            mode = '.'
        elif key == ord('/'):
            mode = '/'
        elif key == ord(';'):
            mode = ';'

        # spatial kernel
        elif key == ord('p'):
            if ksize < 15:
                ksize += 2
        elif key == ord('m'):
            if ksize > 3:
                ksize -= 2

        # canny thresholds
        elif key == ord('i'):
            if high_t < 245:
                high_t += 20
                if high_t <= low_t:
                    low_t = max(5, high_t - 20)
        elif key == ord('k'):
            if high_t > 30:
                high_t -= 20
                if low_t >= high_t:
                    low_t = max(5, high_t - 20)
        elif key == ord('j'):
            if low_t < high_t - 20:
                low_t += 20
        elif key == ord('n'):
            if low_t > 5:
                low_t -= 20

        # freq D0 control
        elif key == ord('z'):
            D0 = min(500, D0 + 5)
        elif key == ord('a'):
            D0 = max(1, D0 - 5)

        # band width control
        elif key == ord('u'):
            band_W = min(400, band_W + 5)
        elif key == ord('d'):
            band_W = max(1, band_W - 5)

        # butterworth order control
        elif key == ord('r'):
            order_n = min(10, order_n + 1)
        elif key == ord('f'):
            order_n = max(1, order_n - 1)

        # contraharmonic q
        elif key == ord('['):
            contra_q = max(-10.0, contra_q - 0.5)
        elif key == ord(']'):
            contra_q = min(10.0, contra_q + 0.5)

        # alpha trimmed d (use \ to -, ' to +)
        elif key == ord('\\'):
            alpha_d = max(0, alpha_d - 2)
        elif key == ord('\''):
            alpha_d = min(40, alpha_d + 2)

        # freq filters (mode becomes 'f')
        elif key == ord('1'):
            freq_filter = 'ILPF'; mode = 'f3'
        elif key == ord('2'):
            freq_filter = 'GLPF'; mode = 'f'
        elif key == ord('3'):
            freq_filter = 'BLPF'; mode = 'f'
        elif key == ord('4'):
            freq_filter = 'IHPF'; mode = 'f'
        elif key == ord('5'):
            freq_filter = 'GHPF'; mode = 'f'
        elif key == ord('6'):
            freq_filter = 'BHPF'; mode = 'f'
        elif key == ord('7'):
            freq_filter = 'IBRF'; mode = 'f'
        elif key == ord('8'):
            freq_filter = 'GBRF'; mode = 'f'
        elif key == ord('9'):
            freq_filter = 'BBRF'; mode = 'f'
        elif key == ord('0'):
            freq_filter = 'IBPF'; mode = 'f'
        elif key == ord('-'):
            freq_filter = 'GBPF'; mode = 'f'
        elif key == ord('='):
            freq_filter = 'BBPF'; mode = 'f'

    cap.release()
    cv2.destroyAllWindows()

webcam_edge_freq_view()
