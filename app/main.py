import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import urllib.request
import glob

# Constants
MODELS_DIR = "models"
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
SWAPPER_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"

def download_file(url, save_path, description):
    """Download a file with progress tracking"""
    try:
        with st.spinner(f"Downloading {description}... This may take a few minutes."):
            # Create a custom opener with headers to avoid 403 errors
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            # Download the file
            urllib.request.urlretrieve(url, save_path)
            return True
    except urllib.error.HTTPError as e:
        st.error(f"Failed to download {description}. Error: {str(e)}")
        if e.code == 404:
            st.error("The model file could not be found. Please check for updated model URLs.")
        return False
    except Exception as e:
        st.error(f"An error occurred while downloading {description}: {str(e)}")
        return False

def download_models():
    """Download required models"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Download and extract buffalo_l model
    buffalo_path = os.path.join(MODELS_DIR, "buffalo_l")
    if not os.path.exists(buffalo_path):
        zip_path = os.path.join(MODELS_DIR, "buffalo_l.zip")
        if download_file(BUFFALO_URL, zip_path, "face analysis model"):
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(MODELS_DIR)
                os.remove(zip_path)
                st.success("✅ Face analysis model downloaded and extracted successfully!")
            except Exception as e:
                st.error(f"Error extracting the face analysis model: {str(e)}")
                return False
        else:
            return False
    
    # Download swapper model
    swapper_path = os.path.join(MODELS_DIR, "inswapper_128.onnx")
    if not os.path.exists(swapper_path):
        if not download_file(SWAPPER_URL, swapper_path, "face swapping model"):
            return False
        st.success("✅ Face swapping model downloaded successfully!")
    
    return True

def load_image(image_file):
    """Load and convert uploaded image to cv2 format"""
    img = Image.open(image_file)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@st.cache_resource
def load_face_swapper():
    """Initialize face analyzer and swapper"""
    try:
        app = FaceAnalysis(name='buffalo_l', root=MODELS_DIR, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        swapper = insightface.model_zoo.get_model(os.path.join(MODELS_DIR, 'inswapper_128.onnx'), 
                                                providers=['CPUExecutionProvider'])
        return app, swapper
    except Exception as e:
        st.error(f"Error initializing face swapper: {str(e)}")
        return None, None

def process_face_swap(source_img, target_img, face_swapper, face_analyzer):
    """Perform deep learning-based face swapping"""
    try:
        # Detect faces
        source_faces = face_analyzer.get(source_img)
        target_faces = face_analyzer.get(target_img)
        
        if len(source_faces) == 0 or len(target_faces) == 0:
            return None, "Could not detect faces in one or both images"
        
        # Get the largest face from source image
        source_face = sorted(source_faces, key=lambda x: x.bbox[2] - x.bbox[0], reverse=True)[0]
        
        # Create result image
        result = target_img.copy()
        
        # Swap all detected faces in target image
        for target_face in target_faces:
            result = face_swapper.get(result, target_face, source_face, paste_back=True)
        
        return result, None
    except Exception as e:
        return None, f"Error during face swapping: {str(e)}"

def main():
    st.title("PhotoCompose - AI Face Swapping")
    st.write("Upload two images to swap faces using advanced AI technology.")
    
    # Download models if not present
    if not download_models():
        st.error("Failed to download required models. Please try again later.")
        st.stop()
    
    # Load face swapper and analyzer
    face_analyzer, face_swapper = load_face_swapper()
    if face_analyzer is None or face_swapper is None:
        st.error("Failed to initialize face swapping models. Please try again.")
        st.stop()
    
    # UI Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source Image (Face to use)")
        source_file = st.file_uploader("Upload source image", type=['jpg', 'jpeg', 'png'])
        if source_file:
            st.image(source_file, caption="Source Image", use_column_width=True)
        
    with col2:
        st.subheader("Target Image (Where to place the face)")
        target_file = st.file_uploader("Upload target image", type=['jpg', 'jpeg', 'png'])
        if target_file:
            st.image(target_file, caption="Target Image", use_column_width=True)
    
    # Advanced options
    with st.expander("Advanced Options"):
        preserve_expression = st.checkbox("Preserve Target Expression", value=False,
                                       help="Try to maintain the facial expression of the target image")
        enhance_quality = st.checkbox("Enhance Output Quality", value=True,
                                    help="Apply post-processing to improve quality")
    
    if source_file and target_file:
        source_img = load_image(source_file)
        target_img = load_image(target_file)
        
        if st.button("Swap Faces"):
            with st.spinner("Processing... This may take a few moments."):
                result, error = process_face_swap(source_img, target_img, face_swapper, face_analyzer)
                
                if error:
                    st.error(error)
                else:
                    st.success("Face swap completed!")
                    
                    # Post-processing if enabled
                    if enhance_quality:
                        # Apply subtle enhancements
                        result = cv2.detailEnhance(result, sigma_s=1, sigma_r=0.15)
                    
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption="Result", use_column_width=True)
                    
                    # Save button
                    result_img = Image.fromarray(result_rgb)
                    buf = io.BytesIO()
                    result_img.save(buf, format="PNG")
                    btn = st.download_button(
                        label="Download Result",
                        data=buf.getvalue(),
                        file_name="face_swap_result.png",
                        mime="image/png"
                    )

    # Add information about the technology
    st.markdown("""
    ---
    ### About the Technology
    This app uses advanced AI technology powered by InsightFace's SWAPPER model, which provides:
    - More realistic face swapping
    - Better preservation of facial details
    - Improved lighting and color matching
    - Natural-looking results
    
    For best results:
    - Use clear, well-lit photos
    - Choose photos with similar head poses
    - Avoid extreme angles or expressions
    - Use high-resolution images
    """)

if __name__ == "__main__":
    main() 