import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import shutil
import pandas as pd
import re

# ==========================
# Configuration
# ==========================
VIDEO_DIR = './videos'       # Folder containing your 12 videos
OUTPUT_DIR = './crops'       # Where to save crops
EMB_DIR = './embeddings'     # Where to save embeddings
MIN_CROPS = 10               # Minimum number of crops per ID to keep
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

# ==========================
# Load ReID and Detection/Tracking Models
# ==========================
PAT_REPO = './Part-Aware-Transformer'
CHECKPOINT_PATH = './Part-Aware-Transformer/part_attention_vit_50.pth'
CONFIG_PATH = './Part-Aware-Transformer/config/PAT.yml'

import sys
sys.path.append(PAT_REPO)
from model.make_model import make_model
from config import cfg

def setup_cfg(config_path):
    import yaml
    with open(config_path, 'r') as file:
        yaml_cfg = yaml.safe_load(file)
    cfg.merge_from_file(config_path)
    return cfg

def load_pat_model(config_path, checkpoint_path, device):
    cfg = setup_cfg(config_path)
    model = make_model(cfg, modelname='part_attention_vit', num_class=751)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    from collections import OrderedDict
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

print("Loading PAT model...")
pat_model = load_pat_model(CONFIG_PATH, CHECKPOINT_PATH, DEVICE)

from ultralytics import YOLO
yolo_model = YOLO("yolo11n.pt")
yolo_model.to(DEVICE)

def yolo_person_tracks(video_path):
    results = yolo_model.track(
        video_path, persist=True, tracker="bytetrack.yaml", classes=[0], 
        verbose=False, save=False, show=False
    )
    from collections import defaultdict
    tracks = defaultdict(list)
    for frame_idx, r in enumerate(results):
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            continue
        for i in range(len(boxes.id)):
            tid = int(boxes.id[i].item())
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            tracks[tid].append((frame_idx, (x1, y1, x2, y2), None))
    return tracks

def get_crops_from_video(tracks, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_dict = {}
    crops_per_track = {}
    required_frames = set([f for tid, track in tracks.items() for f, _, _ in track])
    for frame_idx in sorted(required_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        frame_dict[frame_idx] = frame
    for tid, track in tracks.items():
        crops_per_track[tid] = []
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

# ==========================
# Filename parsing helpers
# ==========================

def parse_station_and_direction(filename):
    # Example: station3_in.mp4 or station12_out.avi
    match = re.match(r"station(\d+)_(in|out)", filename)
    if match:
        station = int(match.group(1))
        direction = match.group(2)
        return station, direction
    else:
        return None, None

# ==========================
# Main Processing Loop
# ==========================

all_rows = []

video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))])
assert len(video_files) == 12, f"Expected 12 videos, found {len(video_files)}."

for video_fname in tqdm(video_files, desc="Videos"):
    video_path = os.path.join(VIDEO_DIR, video_fname)
    station, direction = parse_station_and_direction(video_fname)
    assert station is not None and direction is not None, f"Filename {video_fname} does not match expected format."

    # Save crops in: OUTPUT_DIR/{direction}/station_{station}/id_{tid}/frame_{frame_idx}.jpg
    video_out_dir = os.path.join(OUTPUT_DIR, direction, f"station_{station}")
    os.makedirs(video_out_dir, exist_ok=True)
    print(f"Processing {video_fname} (station={station}, direction={direction})")

    # 1. Track people
    tracks = yolo_person_tracks(video_path)
    # 2. Extract crops
    crops_per_track = get_crops_from_video(tracks, video_path)
    # 3. Remove IDs with fewer than MIN_CROPS crops
    valid_ids = [tid for tid, crops in crops_per_track.items() if len(crops) >= MIN_CROPS]
    for tid in list(crops_per_track.keys()):
        if tid not in valid_ids:
            del crops_per_track[tid]
    # 4. Save crops and prepare for embedding extraction
    for tid, crops in crops_per_track.items():
        id_out_dir = os.path.join(video_out_dir, f"id_{tid}")
        os.makedirs(id_out_dir, exist_ok=True)
        for frame_idx, bbox, crop_img in crops:
            crop_path = os.path.join(id_out_dir, f"frame_{frame_idx}.jpg")
            crop_img.save(crop_path)
            all_rows.append({
                'video_name': video_fname,
                'station': station,
                'direction': direction,
                'id': tid,
                'frame_idx': frame_idx,
                'crop_path': crop_path
            })
    # 5. Remove folders for IDs with < MIN_CROPS (redundant here, but safe)
    for id_folder in os.listdir(video_out_dir):
        id_path = os.path.join(video_out_dir, id_folder)
        imgs = [f for f in os.listdir(id_path) if f.endswith('.jpg')]
        if len(imgs) < MIN_CROPS:
            shutil.rmtree(id_path)

# ==========================
# Remove CSV rows for deleted IDs
# ==========================

df = pd.DataFrame(all_rows)
def keep_row(row):
    return os.path.exists(os.path.dirname(row['crop_path']))
df = df[df.apply(keep_row, axis=1)].reset_index(drop=True)

# ==========================
# Compute and Save Embeddings
# ==========================

def get_embedding(img_path, model, device):
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess_image(img).to(device)
    with torch.no_grad():
        feat = model(img_tensor)
        if isinstance(feat, tuple):
            feat = feat[0]
        emb = feat.cpu().numpy().flatten()
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embeddings"):
    emb_path = os.path.join(
        EMB_DIR,
        f"emb_{row['direction']}_station{row['station']}_id{row['id']}_frame{row['frame_idx']}.npy"
    )
    emb = get_embedding(row['crop_path'], pat_model, DEVICE)
    np.save(emb_path, emb)
    df.at[idx, 'embedding_path'] = emb_path

# ==========================
# Save Mapping CSV
# ==========================
df.to_csv("crop_id_mapping.csv", index=False)
print("Saved crop/ID mapping to crop_id_mapping.csv")
print(f"Saved crops to {OUTPUT_DIR}, embeddings to {EMB_DIR}")