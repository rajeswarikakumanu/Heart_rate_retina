# ==============================
# Heart Rate Detection Config
# ==============================

# Webcam
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Eye Detection
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# MediaPipe Face Mesh
MAX_NUM_FACES = 1

# Blink Detection
EAR_THRESHOLD = 0.22
EAR_CONSEC_FRAMES = 2

# Pulse Signal Buffer
BUFFER_SIZE = 300          # ~10 seconds @ 30 FPS
FPS_ESTIMATE = 30

# Bandpass Filter (Heart Rate Range)
LOWCUT = 0.75             # 45 BPM
HIGHCUT = 3.0             # 180 BPM
FILTER_ORDER = 4

# Heart Rate Limits
MIN_BPM = 45
MAX_BPM = 180

# ROI
ROI_PADDING = 5

# Display
FONT_SCALE = 0.7
LINE_THICKNESS = 2

# Smoothing
HR_SMOOTHING_WINDOW = 5

# Eye Visibility
EYE_VISIBILITY_THRESHOLD = 0.70

# Dashboard Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)