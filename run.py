import streamlit as st
import torch
import sys
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from collections import OrderedDict, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ====== YOLOv8 Imports ======
from ultralytics import YOLO

# ====== SET THESE PATHS FOR YOUR SETUP ======
PAT_REPO = './Part-Aware-Transformer'
CHECKPOINT_PATH = './Part-Aware-Transformer/part_attention_vit_50.pth'
CONFIG_PATH = './Part-Aware-Transformer/config/PAT.yml'

sys.path.append(PAT_REPO)
from model.make_model import make_model
from config import cfg

def setup_cfg(config_path):
    import yaml
    with open(config_path, 'r') as file:
        yaml_cfg = yaml.safe_load(file)
    cfg.merge_from_file(config_path)
    return cfg

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

def preprocess_image(img):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    return transform(img).unsqueeze(0)

@st.cache_resource
def load_yolov8_model():
    model = YOLO("yolov8n.pt")
    model.to("cuda")  # Explicitly move model to GPU

    # Tiny model for speed; for best results use yolov8m or yolov8l if GPU is available
    return model

# ========== 1. Video Uploads ==========
st.title("Entrance-Exit Video Person Matching (YOLOv8+ByteTrack + PAT)")
st.markdown(
    """
    Upload entrance and exit videos (default is 4 of each, or add more). The app uses YOLOv8 with ByteTrack for person tracking and computes a robust identity embedding for each track, then matches entrance and exit IDs.
    """, unsafe_allow_html=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: `{device}`")

NUM_ENTRANCE = st.number_input("Number of entrance videos", min_value=1, max_value=10, value=4, key="num_ent")
NUM_EXIT = st.number_input("Number of exit videos", min_value=1, max_value=10, value=4, key="num_exit")

st.subheader("Upload Entrance Videos")
entrance_videos = st.file_uploader(
    "Select entrance videos", type=['mp4','avi','mov'],
    accept_multiple_files=True, key="entrance_vids"
)
st.subheader("Upload Exit Videos")
exit_videos = st.file_uploader(
    "Select exit videos", type=['mp4','avi','mov'],
    accept_multiple_files=True, key="exit_vids"
)

# ========== 2. YOLOv8+ByteTrack Person Tracking (cached) ==========
def get_crops_from_video(tracks, video_path):
    """Extract crops per track from video using stored bboxes."""
    cap = cv2.VideoCapture(video_path)
    frame_dict = {}
    crops_per_track = defaultdict(list)
    # Collect all required frames
    required_frames = set([f for tid, track in tracks.items() for f, _, _ in track])
    for frame_idx in sorted(required_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        frame_dict[frame_idx] = frame
    for tid, track in tracks.items():
        for frame_idx, bbox, _ in track:
            frame = frame_dict.get(frame_idx)
            if frame is None: continue
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            crop_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crops_per_track[tid].append((frame_idx, bbox, crop_img))
    cap.release()
    return crops_per_track

@st.cache_data(show_spinner=True)
def yolo_person_tracks(video_bytes):
    """Run YOLOv8+ByteTrack on a video (for person class only), return tracks: {track_id: [(frame_idx, bbox, None)]}"""
    # Save video to temp file for YOLO
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tf.write(video_bytes)
    tf.close()
    yolo = load_yolov8_model()
    results = yolo.track(tf.name, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False, save=False, show=False)
    os.unlink(tf.name)
    # Parse YOLOv8 + ByteTrack results
    tracks = defaultdict(list)  # {track_id: [(frame_idx, bbox, None)]}
    for frame_idx, r in enumerate(results):
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            continue
        for i in range(len(boxes.id)):
            tid = int(boxes.id[i].item())
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            tracks[tid].append((frame_idx, (x1, y1, x2, y2), None))
    return tracks

# ========== 3. Compute Centroid Embedding per ID ==========
def compute_id_centroids(crops_per_track, model, device):
    id_embs = {}
    for tid, appearances in crops_per_track.items():
        crops = [img for _, _, img in appearances]
        if not crops: continue
        embs = []
        for crop in crops:
            img_tensor = preprocess_image(crop).to(device)
            with torch.no_grad():
                feat = model(img_tensor)
                if isinstance(feat, tuple): feat = feat[0]
                emb = feat.cpu().numpy().flatten()
            embs.append(emb)
        id_embs[tid] = np.mean(embs, axis=0)
    return id_embs

# ========== 4. Process Videos and Compute Matches ==========
if entrance_videos and exit_videos and len(entrance_videos) == NUM_ENTRANCE and len(exit_videos) == NUM_EXIT:
    with st.spinner("Loading PAT model..."):
        model = load_pat_model(CONFIG_PATH, CHECKPOINT_PATH, device)
    st.info("Extracting YOLOv8 tracks & embeddings from entrance videos...")

    entrance_all_ids = []
    entrance_id_vid = []
    entrance_id_to_crop = {}
    for idx, file in enumerate(entrance_videos):
        tracks = yolo_person_tracks(file.read())
        # Get crops (can be slow for long videos; use only every Nth frame for speed if needed)
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf.write(file.getvalue())
        tf.close()
        crops_per_track = get_crops_from_video(tracks, tf.name)
        id_embs = compute_id_centroids(crops_per_track, model, device)
        for tid, emb in id_embs.items():
            entrance_all_ids.append(emb)
            entrance_id_vid.append((idx, tid))
            # Save one crop for display
            if crops_per_track[tid]:
                entrance_id_to_crop[(idx, tid)] = crops_per_track[tid][0][2]
        os.unlink(tf.name)
    st.success(f"Found {len(entrance_all_ids)} IDs in entrance videos.")

    st.info("Extracting YOLOv8 tracks & embeddings from exit videos...")
    exit_all_ids = []
    exit_id_vid = []
    exit_id_to_crop = {}
    for idx, file in enumerate(exit_videos):
        tracks = yolo_person_tracks(file.read())
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf.write(file.getvalue())
        tf.close()
        crops_per_track = get_crops_from_video(tracks, tf.name)
        id_embs = compute_id_centroids(crops_per_track, model, device)
        for tid, emb in id_embs.items():
            exit_all_ids.append(emb)
            exit_id_vid.append((idx, tid))
            if crops_per_track[tid]:
                exit_id_to_crop[(idx, tid)] = crops_per_track[tid][0][2]
        os.unlink(tf.name)
    st.success(f"Found {len(exit_all_ids)} IDs in exit videos.")

    # ===== Similarity Search =====
    st.info("Computing similarities...")
    if entrance_all_ids and exit_all_ids:
        entrance_all_ids = np.vstack(entrance_all_ids)
        exit_all_ids = np.vstack(exit_all_ids)
        def l2n(x): return x / (np.linalg.norm(x, axis=1, keepdims=True)+1e-12)
        entrance_all_ids = l2n(entrance_all_ids)
        exit_all_ids = l2n(exit_all_ids)
        sim_matrix = cosine_similarity(entrance_all_ids, exit_all_ids)
        top_matches = np.argmax(sim_matrix, axis=1)
        top_scores = sim_matrix[np.arange(len(entrance_all_ids)), top_matches]
        # Visualize matches
        st.subheader("Best Exit ID Match for Each Entrance ID:")
        for i, (ent_pair, match_idx, score) in enumerate(zip(entrance_id_vid, top_matches, top_scores)):
            ent_vid, ent_tid = ent_pair
            exit_vid, exit_tid = exit_id_vid[match_idx]
            col1, col2, col3 = st.columns([1, 0.2, 1])
            with col1:
                st.image(entrance_id_to_crop.get((ent_vid, ent_tid)), caption=f"Entrance V{ent_vid+1} - ID{ent_tid}", width=100)
            with col2:
                st.markdown(f"<div style='text-align:center;font-size:2em;'>&#8594;</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;'>Score:<br><b>{score:.4f}</b></div>", unsafe_allow_html=True)
            with col3:
                st.image(exit_id_to_crop.get((exit_vid, exit_tid)), caption=f"Exit V{exit_vid+1} - ID{exit_tid}", width=100)
            st.markdown("---")
        # Full similarity matrix
        with st.expander("Show full similarity matrix"):
            idxs_entrance = [f"EntV{v+1}-ID{tid}" for v, tid in entrance_id_vid]
            idxs_exit = [f"ExitV{v+1}-ID{tid}" for v, tid in exit_id_vid]
            st.dataframe(
                pd.DataFrame(sim_matrix, columns=idxs_exit, index=idxs_entrance)
                .style.background_gradient(cmap='Blues')
            )
else:
    st.info(f"Upload {NUM_ENTRANCE} entrance and {NUM_EXIT} exit videos to begin.")

st.caption("Powered by YOLOv8, ByteTrack, PAT, and Streamlit.")