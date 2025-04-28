import os
import re
import numpy as np

EMB_DIR = 'embeddings'
CENTROID_DIR = 'centroids'
os.makedirs(CENTROID_DIR, exist_ok=True)

# For collecting all embeddings
groups = {}

# Regex to extract info
pattern = re.compile(
    r'emb_(in|out)_station(\d+)_id(\d+)_frame\d+\.npy'
)

# 1. Group embeddings by (direction, station, id)
for fname in os.listdir(EMB_DIR):
    match = pattern.match(fname)
    if not match:
        continue
    direction, station, id_ = match.groups()
    station = int(station)
    id_ = int(id_)
    key = (direction, station, id_)
    if key not in groups:
        groups[key] = []
    groups[key].append(os.path.join(EMB_DIR, fname))

# 2. Compute and save centroids
centroids = {}
for key, paths in groups.items():
    embs = []
    for path in paths:
        embs.append(np.load(path))
    centroid = np.mean(embs, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    direction, station, id_ = key
    out_fname = f'centroid_{direction}_station{station}_id{id_}.npy'
    out_path = os.path.join(CENTROID_DIR, out_fname)
    np.save(out_path, centroid)
    centroids[key] = centroid

# 3. Prepare entrance and exit lists
entrance_keys = [k for k in centroids if k[0] == 'in']
exit_keys = [k for k in centroids if k[0] == 'out']

# 4. Compute similarity matrix (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

sim_matrix = np.zeros((len(entrance_keys), len(exit_keys)))
for i, in_key in enumerate(entrance_keys):
    for j, out_key in enumerate(exit_keys):
        sim_matrix[i, j] = cosine_similarity(centroids[in_key], centroids[out_key])

# 5. For each entrance, get the top-1 matching exit
print("Entrance\tTop Exit\tSimilarity")
for i, in_key in enumerate(entrance_keys):
    j = np.argmax(sim_matrix[i])
    out_key = exit_keys[j]
    sim = sim_matrix[i, j]
    print(f"station{in_key[1]}_id{in_key[2]} -> station{out_key[1]}_id{out_key[2]} : {sim:.4f}")

# Optionally, print the similarity matrix
print("\nSimilarity Matrix:")
print("Rows: entrances, Columns: exits")
print(sim_matrix)