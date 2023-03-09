from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional

from anndata import AnnData
from ome_zarr.types import LayerData
import numpy as np


def set_id(point_index: int, track_id: int, points: List[Dict]) -> None:
    """Recursively set the ID of all points in the same track."""
    point = points[point_index]
    point["id"] = track_id
    # if we have 1 child and it only has 1 parent, it is part of this track
    if len(point["children"]) == 1:
        child = points[point["children"][0]]
        if len(child["parents"]) == 1:
            set_id(point["children"][0], track_id, points)


def anndata_to_napari_tracks(anndata_obj: AnnData) -> LayerData:
    points_coords = anndata_obj.X
    # convert sparse obsp to dense array
    tracks_matrix = anndata_obj.obsp["tracking"].toarray()

    # we add an 'track_ids' column
    row_count = points_coords.shape[0]

    # for each point, we want to track links to parent/child points
    point_links = []
    for r in range(row_count):
        point_links.append({"parents": [], "children": [], "id": -1})

    # populate all links from the matrix
    for row_index, row in enumerate(tracks_matrix):
        for child_index, value in enumerate(row):
            if value == 1:
                # create a link in both directions...
                point_links[row_index]["children"].append(child_index)
                point_links[child_index]["parents"].append(row_index)

    # Now we can assign track IDs...
    track_id = 0
    for point_index, point in enumerate(point_links):
        # if the ID hasn't been set yet, it's a new track...
        if point["id"] == -1:
            # recursively set ids for all points
            set_id(point_index, track_id, point_links)
            track_id += 1

    # add track IDs as extra column to points, to create 'tracks'
    track_ids = [point["id"] for point in point_links]
    track_ids = np.asarray(track_ids)
    track_ids.resize([row_count, 1])
    tracks = np.concatenate((track_ids, points_coords), axis=1)

    # graph (dict {int: list}) Graph representing associations between tracks.
    # Dictionary defines mapping between a track ID and parents of the track.
    # This can be one (the track has one parent, and the parent has >=1 child)
    # in the case of track splitting, or more than one (the track has multiple
    # parents, but only one child) in the case of track merging.
    graph = defaultdict(list)
    # build a graph of parent links
    for point in point_links:
        for pid in point["parents"]:
            parent = point_links[pid]
            if parent["id"] != point["id"]:
                graph[point["id"]].append(parent["id"])

    # read other properties from obs...
    obs = anndata_obj.obs

    # Properties for each point. Each property should be an array of length N,
    # where N is the number of points.
    # (dict {str: array (N,)}, DataFrame)
    properties = {}
    for colname in obs.columns:
        properties[colname] = obs[colname].values.tolist()

    # The 'arboretum' napari plugin requires each 'properties' to have a 't'
    # https://github.com/quantumjot/BayesianTracker/issues/210
    properties["t"] = points_coords[:, 1]

    print("tracks", tracks)
    print("graph", graph)

    return tracks, {"properties": properties, "graph": graph}, "tracks"


def anndata_to_napari_points(anndata_obj: AnnData) -> LayerData:
    new_layer_data = (
        anndata_obj.X,
        {"edge_width": 0.0, "size": 1, "properties": anndata_obj.obs},
        "points",
    )
    return new_layer_data
