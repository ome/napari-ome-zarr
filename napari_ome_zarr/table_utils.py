
import numpy as np


def get_child_count(row):
    # count the children for a given point
    ones = [val for val in row if val == 1]
    return len(ones)


def get_child_index(row):
    # get the first index of a 1 in the row
    return(row.tolist().index(1))


def track_obj(tracks_matrix, parent_index, points_coords, tracks, track_id):
    row = tracks_matrix[parent_index]
    child_count = get_child_count(row)
    for child_index in range(len(row)):
        if row[child_index] == 1:
            # unset the flag, so we ignore it on next pass
            row[child_index] = 2
            # if we have branching, each new track gets a new ID
            if child_count > 1:
                track_id += 1
            # add row to tracks
            tracks = np.append(tracks, np.insert(points_coords[parent_index], 0, track_id))
            # recursive - the child_index becomes the parent_index
            tracks, track_id = track_obj(tracks_matrix, child_index, points_coords, tracks, track_id)

    return tracks, track_id

              
def anndata_to_napari_tracks(anndata_obj):
    points_coords = anndata_obj.X

    # we add an 'index' column, so that after these points have been
    # reordered into tracks, we can still lookup the obs rows for each point
    row_count = points_coords.shape[0]
    row_ids = np.arange(row_count).reshape([row_count, 1])
    points_coords = np.concatenate((points_coords, row_ids), axis=1)

    # convert sparse obsp to dense array
    tracks_matrix = anndata_obj.obsp["tracking"].toarray()

    # columns: track_id, t, z, y, x, row_index (point_id)
    tracks = np.empty((0, 6))
    track_id = 0
    
    for row_index in range(len(tracks_matrix)):
        # for row_index in range(2):
        for col_index in range(len(tracks_matrix[row_index])):
            if tracks_matrix[row_index][col_index] == 1:
                # for each "1" in sparse matrix row, we add to tracks
                # insert the track_id at the start of point coords, and add to tracks
                tracks = np.append(tracks, np.insert(points_coords[row_index], 0, track_id))
                # recursively track this object...
                tracks, track_id = track_obj(tracks_matrix, col_index, points_coords, tracks, track_id)
                # increment the track id...
                track_id += 1

    tracks = tracks.reshape([tracks.size // 6, 6])

    # read other properties from obs...
    obs = anndata_obj.obs
    obs_dict = obs.to_dict(orient='records')

    # Properties for each point. Each property should be an array of length N, where N is the number of points.
    # (dict {str: array (N,)}, DataFrame)
    properties = {}
    for colname in obs.columns:
        properties[colname] = []

    # The 'arboretum' napari plugin requires each 'properties' to have a 't'
    # https://github.com/quantumjot/BayesianTracker/issues/210
    properties["t"] = []

    # for each row of the tracks data, we get the row_index to add additional data from obs
    for track_row in tracks:
        row_index = int(track_row[5])
        obs_row = obs_dict[row_index]
        properties["t"].append(points_coords[row_index][1])
        for colname in obs.columns:
            properties[colname].append(obs_row[colname])

    # remove the row-index (last column)
    tracks = tracks[:,:5]

    return tracks, {"properties": properties}, 'tracks'


def anndata_to_napari_points(anndata_obj):
    new_layer_data = (
        anndata_obj.X,
        {"edge_width": 0.0, "size": 1, "properties": anndata_obj.obs},
        "points",
    )
    return new_layer_data
