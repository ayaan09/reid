import os
import shutil
import pandas as pd

# Configuration
CROPS_DIR = './crops'
EMB_DIR = './embeddings'
MAPPING_CSV = 'crop_id_mapping.csv'
MERGE_CSV = 'merge_id.csv'
NEW_MAPPING_CSV = 'crop_id_mapping_merged.csv'

# Load mapping CSV and merge/delete plan
df = pd.read_csv(MAPPING_CSV)
merge_map = pd.read_csv(MERGE_CSV)

# Build lookup: (station, direction, old_id) -> new_id
merge_dict = {}
for _, row in merge_map.iterrows():
    key = (int(row['station']), row['direction'], int(row['old_id']))
    new_id = int(row['new_id']) if pd.notnull(row['new_id']) else -1
    merge_dict[key] = new_id

# Prepare for output
new_rows = []

for idx, row in df.iterrows():
    station = int(row['station'])
    direction = row['direction']
    old_id = int(row['id'])
    key = (station, direction, old_id)

    # Determine new_id: use mapping if present, otherwise keep original
    if key in merge_dict:
        new_id = merge_dict[key]
        if new_id == -1:
            # Delete: remove crop and embedding files if they exist, and skip
            if os.path.exists(row['crop_path']):
                os.remove(row['crop_path'])
            if isinstance(row['embedding_path'], str) and os.path.exists(row['embedding_path']):
                os.remove(row['embedding_path'])
            continue
    else:
        new_id = old_id

    # New folder: crops/in/station_X/id_Y/
    new_crop_dir = os.path.join(CROPS_DIR, direction, f"station_{station}", f"id_{new_id}")
    os.makedirs(new_crop_dir, exist_ok=True)
    new_crop_path = os.path.join(new_crop_dir, os.path.basename(row['crop_path']))
    if row['crop_path'] != new_crop_path:
        # If the file already exists (due to merge), you may wish to rename or skip
        if os.path.exists(new_crop_path):
            # To avoid overwriting, you could add a suffix or just skip
            print(f"Warning: {new_crop_path} exists, skipping duplicate frame.")
            continue
        shutil.move(row['crop_path'], new_crop_path)

    # Embedding file: emb_{direction}_station{station}_id{new_id}_frame{frame_idx}.npy
    emb_name = f"emb_{direction}_station{station}_id{new_id}_frame{row['frame_idx']}.npy"
    new_emb_path = os.path.join(EMB_DIR, emb_name)
    if row['embedding_path'] != new_emb_path:
        # Avoid overwriting
        if os.path.exists(new_emb_path):
            print(f"Warning: {new_emb_path} exists, skipping duplicate frame.")
            continue
        shutil.move(row['embedding_path'], new_emb_path)

    # Save updated row
    new_row = row.copy()
    new_row['id'] = new_id
    new_row['crop_path'] = new_crop_path
    new_row['embedding_path'] = new_emb_path
    new_rows.append(new_row)

# Optional: Re-index IDs to be consecutive within each (station, direction)
# (If you want global consecutive IDs, let me know)
# This section is optional and can be omitted if you want to keep your assigned new_id numbers.
# If you want to re-index, uncomment the following block:

# from collections import defaultdict
# station_dir_ids = defaultdict(set)
# for r in new_rows:
#     station_dir_ids[(r['station'], r['direction'])].add(r['id'])
# id_maps = {}
# for (station, direction), ids in station_dir_ids.items():
#     ids = sorted(ids)
#     id_maps[(station, direction)] = {old_id: idx+1 for idx, old_id in enumerate(ids)}
# for r in new_rows:
#     key = (r['station'], r['direction'])
#     r['id'] = id_maps[key][r['id']]
#     # Update paths accordingly (not shown for brevity, but you can adapt above)

# Save new mapping CSV
new_df = pd.DataFrame(new_rows)
new_df.to_csv(NEW_MAPPING_CSV, index=False)
print(f"Saved new mapping to {NEW_MAPPING_CSV}")

# Clean up empty directories (optional)
def clean_empty_dirs(root):
    for path, dirs, files in os.walk(root, topdown=False):
        if not os.listdir(path):
            os.rmdir(path)
clean_empty_dirs(CROPS_DIR)
clean_empty_dirs(EMB_DIR)