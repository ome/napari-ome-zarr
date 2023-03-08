
import numpy as np


def get_child_count(row):
    # count the children for a given point
    ones = [val for val in row if val == 1]
    return len(ones)

              
def anndata_to_napari_tracks(anndata_obj):
    points_coords = anndata_obj.X
    # convert sparse obsp to dense array
    tracks_matrix = anndata_obj.obsp["tracking"].toarray()

    # graph (dict {int: list}) – Graph representing associations between tracks. Dictionary defines the mapping between a track ID and the parents of the track. This can be one (the track has one parent, and the parent has >=1 child) in the case of track splitting, or more than one (the track has multiple parents, but only one child) in the case of track merging. See examples/tracks_3d_with_graph.py
    graph = {}

    # we add an 'track_ids' column
    row_count = points_coords.shape[0]
    track_ids = np.zeros([row_count, 1])

    track_id = 0
    # track_ids[0] = track_id
    for row_index in range(len(tracks_matrix)):
        if track_ids[row_index] == 0:
            track_ids[row_index] = track_id
        row = tracks_matrix[row_index]
        child_count = get_child_count(row)
        print('row_index', row_index, 'child_count', child_count, 'track_id', track_id)
        # end of track - increment for next...
        if child_count == 0:
            track_id += 1
        for child_index in range(len(row)):
            if row[child_index] == 1:
                if child_count == 1:
                    # orphan child will have the same track_id as parent
                    if track_ids[child_index] == 0:
                        track_ids[child_index] = track_ids[row_index]
                    else:
                        # If the child was already a child, this is a new merged track
                        track_id += 1
                        track_ids[child_index] = track_id
                else:
                    # multiple child tracks, each with new track_id
                    track_id += 1
                    track_ids[child_index] = track_id

    tracks = np.concatenate((track_ids, points_coords), axis=1)


    # read other properties from obs...
    obs = anndata_obj.obs
    obs_dict = obs.to_dict(orient='records')

    # Properties for each point. Each property should be an array of length N, where N is the number of points.
    # (dict {str: array (N,)}, DataFrame)
    properties = {}
    for colname in obs.columns:
        properties[colname] = obs[colname].values.tolist()

    # The 'arboretum' napari plugin requires each 'properties' to have a 't'
    # https://github.com/quantumjot/BayesianTracker/issues/210
    properties["t"] = points_coords[:,1]

    print("tracks", tracks)

    return tracks, {"properties": properties}, 'tracks'


def anndata_to_napari_points(anndata_obj):
    new_layer_data = (
        anndata_obj.X,
        {"edge_width": 0.0, "size": 1, "properties": anndata_obj.obs},
        "points",
    )
    return new_layer_data
