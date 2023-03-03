
import numpy as np


def check_div(tracks_matrix, n):
    next = 0
    for i in range(len(tracks_matrix[n])):
            if tracks_matrix[n][i] == 1:
                next += 1
    return next


def track_obj(tracks_matrix, m, points_coords, tracks, tid):
    if check_div(tracks_matrix, m) == 1:
        tracks = np.append(tracks, np.insert(points_coords[m], 0, tid))
        for i in range(len(tracks_matrix[m])):
            if tracks_matrix[m][i] == 1:
                tracks_matrix[m][i] = 2
                tracks = track_obj(tracks_matrix, i, points_coords, tracks, tid)
        return tracks
    else:
        return tracks

              
def anndata_to_napari_tracks(anndata_obj):
    points_coords = anndata_obj.X
    # convert sparse obsp to dense array
    tracks_matrix = anndata_obj.obsp["tracking"].toarray()

    tracks = np.empty((0, 5))
    
    tid = 0
    for i in range(len(tracks_matrix)):
        for j in range(len(tracks_matrix[i])):
            if tracks_matrix[i][j] == 1:
                tracks = np.append(tracks, np.insert(points_coords[i], 0, tid))
                tracks = track_obj(tracks_matrix, j, points_coords, tracks, tid)
                tid += 1

    t = tracks.reshape([tracks.size // 5, 5])

    return t, {}, 'tracks'


def anndata_to_napari_points(anndata_obj):
    new_layer_data = (
        anndata_obj.X,
        {"edge_width": 0.0, "size": 1, "properties": anndata_obj.obs},
        "points",
    )
    return new_layer_data
