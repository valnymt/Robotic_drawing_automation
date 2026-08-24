"""Shared constants for the draw-arm pipeline.

Every script imports these instead of redefining them, so the arm
geometry and workspace bounds only ever live in one place.
"""

# --- Arm geometry (DH link lengths, meters) ---
# Matches robotics_theory_reference.md section 2.
L1 = 0.5   # shoulder -> elbow
L2 = 0.4   # elbow -> pen tip

# Reachable ring, derived from L1/L2 (section 4):
#   |L1 - L2| <= r <= L1 + L2  ->  0.10 m <= r <= 0.90 m
R_MIN = abs(L1 - L2)
R_MAX = L1 + L2

# --- Drawing workspace (robot-frame rectangle the arm draws into) ---
# x in [X0, X1], y in [Y0, Y1]. Kept inside the reachable ring above.
X0 = 0.25
X1 = 0.70
Y0 = -0.20
Y1 = 0.20

# --- Vision / image pipeline ---
IMG_TARGET_SIZE = (640, 640)   # (W, H) resize target before edge detection
CANNY_LOW = 50
CANNY_HIGH = 150
GAUSSIAN_BLUR_KSIZE = (5, 5)
APPROX_POLY_EPS_FRAC = 0.01     # fraction of contour arc length, for cv2.approxPolyDP

# --- Trajectory ---
STEPS_PER_SEGMENT = 2

# --- Simulation ---
SIM_TIMESTEP = 1.0 / 240.0
