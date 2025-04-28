import streamlit as st
import torch
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

# ====== SET THESE PATHS FOR YOUR SETUP ======
PAT_REPO = './Part-Aware-Transformer'  # Path to your PAT repo root
CHECKPOINT_PATH = './Part-Aware-Transformer/part_attention_vit_50.pth'  # Trained .pth checkpoint
CONFIG_PATH = './Part-Aware-Transformer/config/PAT.yml'  # Path to your PAT.yml config

# ====== ADD PAT REPO TO PYTHONPATH ======
sys.path.append(PAT_REPO)

# ====== IMPORTS FROM YOUR PAT REPO ======
from model.make_model import make_model
from config import cfg

# ====== CONFIG LOADING ======
def setup_cfg(config_path):
    import yaml
    with open(config_path, 'r') as file:
        yaml_cfg = yaml.safe_load(file)
    cfg.merge_from_file(config_path)
    return cfg

# ====== MODEL LOADING ======
@st.cache_resource
def load_pat_model(config_path, checkpoint_path, device):
    cfg = setup_cfg(config_path)
    model = make_model(cfg, modelname='part_attention_vit', num_class=751)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith('classifier'):
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    return model

# ====== IMAGE PREPROCESSING ======
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    return transform(img).unsqueeze(0)

# ====== STREAMLIT APP ======
st.title("Compare Two Images with PAT Embeddings")
st.markdown(
    "Upload two images. This app will generate embeddings using your trained PAT model and compute their cosine similarity."
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: `{device}`")

uploaded_files = st.file_uploader(
    "Upload exactly two images", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 2:
    images = []
    for i, file in enumerate(uploaded_files):
        img = Image.open(file).convert('RGB')
        st.image(img, caption=f"Image {i+1}", width=200)
        images.append(img)
    with st.spinner("Loading model and extracting embeddings..."):
        model = load_pat_model(CONFIG_PATH, CHECKPOINT_PATH, device)
        embeddings = []
        for img in images:
            img_tensor = preprocess_image(img).to(device)
            with torch.no_grad():
                feat = model(img_tensor)
                if isinstance(feat, tuple):
                    feat = feat[0]
                emb = feat.cpu().numpy().flatten()
            embeddings.append(emb)
    emb1, emb2 = embeddings
    sim = cosine_similarity([emb1], [emb2])[0][0]
    st.success(f"Cosine Similarity: {sim:.4f}")
    st.write("Embedding 1 shape:", emb1.shape)
    st.write("Embedding 2 shape:", emb2.shape)
else:
    st.info("Please upload two images to compare.")

