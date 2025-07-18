import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
import os
from functools import lru_cache
import pandas as pd

st.set_page_config(page_title='LOOKS-MAXXING AI', layout='wide')

st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #121212 !important;
    }
    html, body, [class^="css"]  {
        font-size: 20px !important;
    }
    .stButton > button {
        font-size: 22px !important;
        padding: 1em 2em !important;
        border-radius: 10px !important;
        background-color: #003366 !important;
        color: #fff !important;
        margin-bottom: 1em !important;
    }
    .stFileUploader {
        font-size: 22px !important;
        padding: 1.5em !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

skincare_table = [
    {
        'Skin Concern': 'Acne',
        'Top Ingredients': 'Salicylic Acid, Benzoyl Peroxide, Retinoids',
        'AM Routine': '1. Cleanse: Salicylic Acid or Benzoyl Peroxide Cleanser<br>2. Treat: Niacinamide Serum<br>3. Moisturize: Oil-Free, Non-Comedogenic Moisturizer<br>4. Protect: SPF 30+ (Oil-Free)',
        'PM Routine': '1. Cleanse: Gentle Cleanser<br>2. Treat: Retinoid (Adapalene) or a targeted spot treatment<br>3. Moisturize: Non-Comedogenic Moisturizer',
    },
    {
        'Skin Concern': 'Dark Spots',
        'Top Ingredients': 'Vitamin C, Niacinamide, AHAs (Glycolic Acid)',
        'AM Routine': '1. Cleanse: Gentle Cleanser<br>2. Treat: Vitamin C Serum<br>3. Moisturize: Hydrating Moisturizer<br>4. Protect: SPF 30+ (Crucial!)',
        'PM Routine': '1. Cleanse: Gentle Cleanser<br>2. Treat: AHA/Glycolic Acid Toner or Retinoid Serum<br>3. Moisturize: Moisturizer with Niacinamide',
    },
    {
        'Skin Concern': 'Pores',
        'Top Ingredients': 'Salicylic Acid, Niacinamide, Clay',
        'AM Routine': '1. Cleanse: Salicylic Acid or Clay-based Cleanser<br>2. Treat: Niacinamide Serum<br>3. Moisturize: Lightweight, Gel-based Moisturizer<br>4. Protect: SPF 30+ (Non-Comedogenic)',
        'PM Routine': '1. Cleanse: Gentle Cleanser<br>2. Treat: Salicylic Acid (BHA) Toner or Retinoid<br>3. Moisturize: Lightweight Moisturizer<br>Weekly: Clay Mask',
    },
    {
        'Skin Concern': 'Wrinkles',
        'Top Ingredients': 'Retinoids, Vitamin C, Peptides, Hyaluronic Acid',
        'AM Routine': '1. Cleanse: Hydrating Cleanser<br>2. Treat: Vitamin C & Hyaluronic Acid Serum<br>3. Moisturize: Peptide-rich Cream<br>4. Protect: SPF 30+',
        'PM Routine': '1. Cleanse: Hydrating Cleanser<br>2. Treat: Retinoid Serum<br>3. Moisturize: Rich Moisturizer with Ceramides or Peptides',
    },
    {
        'Skin Concern': 'Redness',
        'Top Ingredients': 'Azelaic Acid, Niacinamide, Ceramides, Centella Asiatica',
        'AM Routine': '1. Cleanse: Creamy, Fragrance-Free Cleanser<br>2. Treat: Niacinamide or Azelaic Acid Serum<br>3. Moisturize: Ceramide-rich Moisturizer<br>4. Protect: Mineral Sunscreen (Zinc/Titanium) SPF 30+',
        'PM Routine': '1. Cleanse: Creamy, Fragrance-Free Cleanser<br>2. Treat: Calming Serum (Centella/Cica)<br>3. Moisturize: Soothing Night Cream with Ceramides',
    },
]

from skin_analysis.acne import detect_acne, detect_acne_ml, load_acne_model
from skin_analysis.oiliness import detect_oiliness
from skin_analysis.uv import detect_uv_exposure
from skin_analysis.dark_spots import detect_dark_spots
from skin_analysis.redness import detect_redness
from skin_analysis.wrinkles import detect_wrinkles
from skin_analysis.pores import detect_pores
from skin_analysis.regions import extract_face_regions

# Title as the first line
# Two-column layout: left for title and instruction, right for logo
col_left, col_right = st.columns([3, 1])
with col_left:
    st.markdown('<div style="font-size:2.5em; font-weight:800; margin-bottom:0.1em;">✨ LOOKSMAXXING AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.26em; font-weight:700; margin-bottom:0.2em; margin-top:0.1em;">Upload a selfie or take a photo to detect skin defects<br>and get personalized skincare tips!</div>', unsafe_allow_html=True)
    # Add two empty lines for spacing before input method
    st.markdown('<div style="height:2.4em;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.56em; font-weight:600; margin-bottom:0.1em;">Choose input method:</div>', unsafe_allow_html=True)
    input_mode = st.radio('', ['Upload a photo', 'Take a selfie'])
    uploaded_file = None
    img_data = None
    if input_mode == 'Upload a photo':
        st.markdown('<div style="font-size:1.43em; margin-bottom:0.1em;">Choose a face image...</div>', unsafe_allow_html=True)
        st.markdown('<style>div[data-testid="stFileUploader"] > label {display: none;}</style>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False, key='file_uploader')
        if uploaded_file is not None:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.warning('File is too large! Please upload an image smaller than 10MB.')
                uploaded_file = None
    else:
        st.markdown('<div id="camera-region"></div>', unsafe_allow_html=True)
        img_data = st.camera_input('Take a selfie')
        st.markdown(
            """
            <script>
            const anchor = document.getElementById('camera-region');
            if (anchor) {
                anchor.scrollIntoView({ behavior: 'smooth' });
            }
            </script>
            """,
            unsafe_allow_html=True
        )
with col_right:
    st.image('logo.png', width=480)

# Place the file uploader further down if in upload mode
# if input_mode == 'Upload a photo':
#     with uploader_placeholder:
#         uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False, key='file_uploader')
#         if uploaded_file is not None:
#             if uploaded_file.size > 10 * 1024 * 1024:
#                 st.warning('File is too large! Please upload an image smaller than 10MB.')
#                 uploaded_file = None

# Helper for recommendations
def get_recommendations(region_results):
    recs = []
    for region, res in region_results.items():
        region_recs = []
        if res['acne'][0]:
            region_recs.append('Acne detected')
        if res['oiliness'] > 50:
            region_recs.append('High oiliness')
        if res['redness'] > 10:
            region_recs.append('Redness')
        if res['dark_spots'] > 2:
            region_recs.append('Dark spots')
        if res['wrinkles'] > 0.05:
            region_recs.append('Wrinkles')
        if res['pores'] > 10:
            region_recs.append('Visible pores')
        if not region_recs:
            region_recs.append('Healthy')
        recs.append(f"{region}: {', '.join(region_recs)}")
    return recs

def draw_region_boxes(img, regions):
    img_draw = img.copy()
    color_map = {
        'T-zone': (255, 0, 0),
        'Left Cheek': (0, 255, 0),
        'Right Cheek': (0, 0, 255),
        'Chin': (255, 255, 0)
    }
    for region, crop in regions.items():
        if crop.size == 0:
            continue
        h, w = crop.shape[:2]
        for y in range(img.shape[0] - h + 1):
            for x in range(img.shape[1] - w + 1):
                if np.array_equal(img[y:y+h, x:x+w], crop):
                    cv2.rectangle(img_draw, (x, y), (x+w, y+h), color_map.get(region, (255,255,255)), 2)
                    cv2.putText(img_draw, region, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map.get(region, (255,255,255)), 2)
                    break
    return img_draw

def highlight_issues(img, region_results, regions):
    img_draw = img.copy()
    for region, res in region_results.items():
        crop = regions.get(region)
        if crop is None or crop.size == 0:
            continue
        h, w = crop.shape[:2]
        for y in range(img.shape[0] - h + 1):
            for x in range(img.shape[1] - w + 1):
                if np.array_equal(img[y:y+h, x:x+w], crop):
                    if res['acne'][0]:
                        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0,0,255), 2)
                    if res['oiliness'] > 50:
                        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0,255,255), 2)
                    if res['redness'] > 10:
                        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0,0,255), 2)
                    if res['dark_spots'] > 2:
                        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (128,0,128), 2)
                    if res['wrinkles'] > 0.05:
                        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (255,0,255), 2)
                    if res['pores'] > 10:
                        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0,255,0), 2)
                    break
    return img_draw

def auto_crop_face(img_cv, cascade_path='haarcascade_frontalface_default.xml'):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img_cv  # fallback: return original
    x, y, w, h = faces[0]
    # Add margin
    margin = int(0.2 * max(w, h))
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, img_cv.shape[1])
    y2 = min(y + h + margin, img_cv.shape[0])
    return img_cv[y1:y2, x1:x2]

def get_acne_coords(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, minDist=10, param1=50, param2=15, minRadius=2, maxRadius=10)
    coords = []
    if circles is not None:
        for c in circles[0]:
            coords.append((int(c[0]), int(c[1])))
    return coords

def get_dark_spot_coords(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coords = []
    for cnt in contours:
        if 10 < cv2.contourArea(cnt) < 200:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                coords.append((cx, cy))
    return coords

def get_pore_coords(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 100
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(blur)
    coords = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]
    return coords

def get_wrinkle_coords(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    ys, xs = np.where(edges > 0)
    coords = list(zip(xs, ys))
    if len(coords) > 30:
        coords = coords[::len(coords)//30]
    return coords

def get_redness_coords(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    ys, xs = np.where(mask > 0)
    coords = list(zip(xs, ys))
    if len(coords) > 30:
        coords = coords[::len(coords)//30]
    return coords

def draw_defect_markers_fullface(face_img):
    img_draw = face_img.copy()
    color_map = {
        'acne': (0, 0, 255),        # Red
        'dark_spots': (0, 255, 0), # Green
        'pores': (255, 0, 0),      # Blue
        'wrinkles': (128, 0, 128), # Purple
        'redness': (0, 140, 255),  # Orange
    }
    acne_pts = get_acne_coords(face_img)
    dark_pts = get_dark_spot_coords(face_img)
    pore_pts = get_pore_coords(face_img)
    wrinkle_pts = get_wrinkle_coords(face_img)
    red_pts = get_redness_coords(face_img)
    for (x, y) in acne_pts:
        cv2.circle(img_draw, (x, y), 7, color_map['acne'], 2)
    for (x, y) in dark_pts:
        cv2.circle(img_draw, (x, y), 7, color_map['dark_spots'], 2)
    for (x, y) in pore_pts:
        cv2.circle(img_draw, (x, y), 5, color_map['pores'], 1)
    for (x, y) in wrinkle_pts:
        cv2.circle(img_draw, (x, y), 3, color_map['wrinkles'], 1)
    for (x, y) in red_pts:
        cv2.circle(img_draw, (x, y), 3, color_map['redness'], 1)
    return img_draw

def draw_defect_markers_regions(face_img, regions):
    img_draw = face_img.copy()
    color_map = {
        'acne': (0, 0, 255),        # Red
        'dark_spots': (0, 255, 0), # Green
        'pores': (255, 0, 0),      # Blue
        'wrinkles': (128, 0, 128), # Purple
        'redness': (0, 140, 255),  # Orange
    }
    for region, crop in regions.items():
        if crop.size == 0:
            continue
        h, w = crop.shape[:2]
        # Find where the crop is in the original image
        found = False
        for y in range(img_draw.shape[0] - h + 1):
            for x in range(img_draw.shape[1] - w + 1):
                if np.array_equal(img_draw[y:y+h, x:x+w], crop):
                    # Get defect coordinates in region
                    acne_pts = get_acne_coords(crop)
                    dark_pts = get_dark_spot_coords(crop)
                    pore_pts = get_pore_coords(crop)
                    wrinkle_pts = get_wrinkle_coords(crop)
                    red_pts = get_redness_coords(crop)
                    for (px, py) in acne_pts:
                        cv2.circle(img_draw, (x+px, y+py), 7, color_map['acne'], 2)
                    for (px, py) in dark_pts:
                        cv2.circle(img_draw, (x+px, y+py), 7, color_map['dark_spots'], 2)
                    for (px, py) in pore_pts:
                        cv2.circle(img_draw, (x+px, y+py), 5, color_map['pores'], 1)
                    for (px, py) in wrinkle_pts:
                        cv2.circle(img_draw, (x+px, y+py), 3, color_map['wrinkles'], 1)
                    for (px, py) in red_pts:
                        cv2.circle(img_draw, (x+px, y+py), 3, color_map['redness'], 1)
                    found = True
                    break
            if found:
                break
    return img_draw

defect_legend = {
    'Acne': 'Red',
    'Dark Spots': 'Green',
    'Pores': 'Blue',
    'Wrinkles': 'Purple',
    'Redness': 'Orange',
}

# Cache ML model loading
@lru_cache(maxsize=1)
def get_acne_model():
    if os.path.exists('acne_model.h5'):
        return load_acne_model('acne_model.h5')
    return None

def extract_face_regions_extended(img: np.ndarray, face_cascade_path='haarcascade_frontalface_default.xml'):
    """
    Detect face and extract T-zone, left cheek, right cheek, chin, nose, left under-eye, right under-eye regions.
    Returns a dict: region name -> image crop.
    """
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    regions = {}
    if len(faces) == 0:
        return regions
    x, y, w, h = faces[0]
    # T-zone: upper center
    tzone = img[y:y+int(0.4*h), x+int(0.3*w):x+int(0.7*w)]
    # Left cheek
    left_cheek = img[y+int(0.4*h):y+int(0.7*h), x:x+int(0.3*w)]
    # Right cheek
    right_cheek = img[y+int(0.4*h):y+int(0.7*h), x+int(0.7*w):x+w]
    # Chin
    chin = img[y+int(0.7*h):y+h, x+int(0.3*w):x+int(0.7*w)]
    # Nose (center vertical strip)
    nose = img[y+int(0.4*h):y+int(0.7*h), x+int(0.45*w):x+int(0.55*w)]
    # Left under-eye
    left_eye = img[y+int(0.25*h):y+int(0.4*h), x+int(0.15*w):x+int(0.35*w)]
    # Right under-eye
    right_eye = img[y+int(0.25*h):y+int(0.4*h), x+int(0.65*w):x+int(0.85*w)]
    regions['T-zone'] = tzone
    regions['Left Cheek'] = left_cheek
    regions['Right Cheek'] = right_cheek
    regions['Chin'] = chin
    regions['Nose'] = nose
    regions['Left Under-eye'] = left_eye
    regions['Right Under-eye'] = right_eye
    return regions

def get_user_concerns(region_results):
    concerns = set()
    for res in region_results.values():
        if res['acne'][0]:
            concerns.add('Acne')
        if res['dark_spots'] > 2:
            concerns.add('Dark Spots')
        if res['pores'] > 10:
            concerns.add('Pores')
        if res['wrinkles'] > 0.05:
            concerns.add('Wrinkles')
        if res['redness'] > 10:
            concerns.add('Redness')
    return sorted(concerns)

def build_personalized_routine(user_concerns, skincare_table):
    morning_steps = []
    evening_steps = []
    ingredients = set()
    for concern in user_concerns:
        for row in skincare_table:
            if row['Skin Concern'] == concern:
                ingredients.update([i.strip() for i in row['Top Ingredients'].split(',')])
                morning_steps.append(f'<b>{concern}:</b><br>' + row['AM Routine'].replace('AM Routine', 'Morning Routine').replace('PM Routine', 'Evening Routine'))
                evening_steps.append(f'<b>{concern}:</b><br>' + row['PM Routine'].replace('AM Routine', 'Morning Routine').replace('PM Routine', 'Evening Routine'))
    return ingredients, morning_steps, evening_steps

# Main logic
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
elif img_data is not None:
    image = Image.open(img_data).convert('RGB')

# Add sidebar for instructions and legend
st.sidebar.title('How to Use')
st.sidebar.markdown('''
1. Upload a photo or take a selfie.
2. Wait for face detection and region extraction.
3. Review detected face regions and defect markers.
4. Click **Run Full Skin Analysis** for detailed results.
''')
st.sidebar.markdown('---')
st.sidebar.info('**For best results:**\nTake your photo in good lighting and ensure your face is clearly visible for accurate analysis.')
st.sidebar.subheader('Defect Marker Legend')
st.sidebar.markdown('<span style="color:#ff0000;">●</span> Acne  \
<span style="color:#00ff00;">●</span> Dark Spots  \
<span style="color:#0000ff;">●</span> Pores  \
<span style="color:#800080;">●</span> Wrinkles  \
<span style="color:#ff8c00;">●</span> Redness', unsafe_allow_html=True)

if image is not None:
    st.markdown('---')
    st.header('Step 1: Face Detection and Region Extraction')
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    with st.spinner('Detecting and cropping face...'):
        cropped_face = auto_crop_face(img_cv)
    st.image(Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)), caption='Auto-cropped Face', use_container_width=True)
    with st.spinner('Extracting face regions...'):
        regions = extract_face_regions_extended(cropped_face)

    st.markdown('---')
    st.header('Step 2: Visualize Detected Regions and Defects')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Detected Face Regions')
        img_regions = draw_region_boxes(cropped_face, regions)
        st.image(Image.fromarray(cv2.cvtColor(img_regions, cv2.COLOR_BGR2RGB)), caption='Face Regions', use_container_width=True)
    with col2:
        st.subheader('Defect Marker Overlay (Face Regions Only)')
        img_defects = draw_defect_markers_regions(cropped_face.copy(), regions)
        st.image(Image.fromarray(cv2.cvtColor(img_defects, cv2.COLOR_BGR2RGB)), caption='Defect Locations (Face Regions)', use_container_width=True)

    st.markdown('---')
    st.header('Step 3: Full Skin Analysis')
    st.markdown('Click the button below to run a detailed analysis and get region-wise results and recommendations.')
    if st.button('Run Full Skin Analysis'):
        with st.spinner('Analyzing skin conditions...'):
            region_results = {}
            model = get_acne_model()
            n_regions = len([crop for crop in regions.values() if crop.size != 0])
            progress = st.progress(0)
            i = 0
            for region, crop in regions.items():
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (128, 128))
                acne = (False, 0.0)
                try:
                    if model is not None:
                        acne = detect_acne_ml(crop_resized)
                    else:
                        acne = detect_acne(crop_resized)
                except Exception:
                    acne = detect_acne(crop_resized)
                oiliness = detect_oiliness(crop_resized)
                uv = detect_uv_exposure(crop_resized)
                dark_spots = detect_dark_spots(crop_resized)
                redness = detect_redness(crop_resized)
                wrinkles = detect_wrinkles(crop_resized)
                pores = detect_pores(crop_resized)
                region_results[region] = {
                    'acne': acne,
                    'oiliness': oiliness,
                    'uv': uv,
                    'dark_spots': dark_spots,
                    'redness': redness,
                    'wrinkles': wrinkles,
                    'pores': pores
                }
                i += 1
                progress.progress(i / n_regions)
            img_issues = highlight_issues(cropped_face, region_results, regions)

        st.markdown('---')
        st.header('Step 4: Results and Recommendations')
        with st.expander('See Region-wise Analysis Table'):
            df = pd.DataFrame({region: {
                'Acne': 'Yes' if res['acne'][0] else 'No',
                'Oiliness %': f"{res['oiliness']:.1f}",
                'Redness %': f"{res['redness']:.1f}",
                'Dark Spots': res['dark_spots'],
                'Wrinkle Score': f"{res['wrinkles']:.2f}",
                'Pores': res['pores']
            } for region, res in region_results.items()}).T
            st.dataframe(df)

        st.subheader('Region-specific Recommendations')
        for rec in get_recommendations(region_results):
            st.write('- ' + rec)

        # Personalized routine
        user_concerns = get_user_concerns(region_results)
        if user_concerns:
            st.markdown('---')
            st.subheader('Your Personalized Skincare Routine')
            st.markdown('**Detected Concerns:** ' + ', '.join(user_concerns))
            ingredients, morning_steps, evening_steps = build_personalized_routine(user_concerns, skincare_table)
            st.markdown('**Top Ingredients to Look For:** ' + ', '.join(sorted(ingredients)))
            st.markdown('')
            col_morning, col_evening = st.columns(2)
            with col_morning:
                st.markdown('<h4 style="margin-bottom:0.5em;">🌞 Morning Routine</h4>', unsafe_allow_html=True)
                for step in morning_steps:
                    st.markdown(f'<div style="margin-bottom:1.5em;">{step}</div>', unsafe_allow_html=True)
            with col_evening:
                st.markdown('<h4 style="margin-bottom:0.5em;">🌙 Evening Routine</h4>', unsafe_allow_html=True)
                for step in evening_steps:
                    st.markdown(f'<div style="margin-bottom:1.5em;">{step}</div>', unsafe_allow_html=True) 