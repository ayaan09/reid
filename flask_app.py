from flask import Flask, render_template, jsonify, request
import os
import re
import numpy as np

app = Flask(__name__)

CENTROID_DIR = "centroids"
VIDEO_DIR = "static/videos"

# Utility: get station list
def get_station_list():
    entrance = set()
    exit_ = set()
    pat = re.compile(r'centroid_(in|out)_station(\d+)_id(\d+)\.npy')
    for fname in os.listdir(CENTROID_DIR):
        m = pat.match(fname)
        if m:
            if m.group(1) == "in":
                entrance.add(int(m.group(2)))
            else:
                exit_.add(int(m.group(2)))
    # For demo, ensure at least 4
    entrance = sorted(entrance | {1,2,3,4})
    exit_ = sorted(exit_ | {1,2,3,4})
    return entrance, exit_

# Utility: get all centroids per station-direction
def get_centroids(direction, station):
    # Returns dict: id -> embedding
    pat = re.compile(rf'centroid_{direction}_station{station}_id(\d+)\.npy')
    d = {}
    for fname in os.listdir(CENTROID_DIR):
        m = pat.match(fname)
        if m:
            arr = np.load(os.path.join(CENTROID_DIR, fname))
            d[int(m.group(1))] = arr
    return d

@app.route("/")
def index():
    entrance_stations, exit_stations = get_station_list()
    return render_template(
        "index.html",
        entrance_list=entrance_stations,
        exit_list=exit_stations,
    )

@app.route("/compute_od", methods=["POST"])
def compute_od():
    data = request.json
    entrance_stations = data["entrance_stations"]
    exit_stations = data["exit_stations"]

    # Gather all entrance and exit centroids by (station, id)
    entrance_people = []  # tuples: (station_idx, id, embedding)
    for en_idx, en_st in enumerate(entrance_stations):
        cents = get_centroids("in", en_st)
        for id_, emb in cents.items():
            entrance_people.append( (en_idx, id_, emb) )
    exit_people = []      # tuples: (station_idx, id, embedding)
    for ex_idx, ex_st in enumerate(exit_stations):
        cents = get_centroids("out", ex_st)
        for id_, emb in cents.items():
            exit_people.append( (ex_idx, id_, emb) )

    # For each entrance person, find most similar exit person
    od = np.zeros((len(exit_stations), len(entrance_stations)), dtype=int)
    if entrance_people and exit_people:
        for en_idx, id1, emb1 in entrance_people:
            best_ex_idx = None
            best_sim = -2.
            for ex_idx, id2, emb2 in exit_people:
                sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1)*np.linalg.norm(emb2) + 1e-12))
                if sim > best_sim:
                    best_sim = sim
                    best_ex_idx = ex_idx
            if best_ex_idx is not None:
                od[best_ex_idx, en_idx] += 1

    return jsonify({
        "od_matrix": od.tolist(),
        "entrance_stations": entrance_stations,
        "exit_stations": exit_stations
    })

if __name__ == "__main__":
    app.run(port=8005, debug=True)