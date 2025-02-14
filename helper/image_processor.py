import moondream as md
from PIL import Image
import streamlit as st

MOONDREAM_API_KEY = st.secrets["MOONDREAM_API_KEY"]

def process_image(image_path):
    # Initialize with API key
    model = md.vl(api_key=MOONDREAM_API_KEY)
    
    # Load an image
    image = Image.open(image_path)
    
    # Generate a caption
    caption = model.caption(image)["caption"]
    print("Caption:", caption)

    return caption