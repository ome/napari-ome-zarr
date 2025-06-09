import dask.array as da
import numpy as np
from zarr import Group


def get_attrs(group: Group):
    if "ome" in group.attrs:
        return group.attrs["ome"]
    return group.attrs


def get_pyramid_lazy(plate_group, labels_path=None) -> None:
    """
    Return a pyramid of dask data, where the highest resolution is the
    stitched full-resolution images.
    """
    # plate_data = plate_group.attrs["plate"]
    # well_paths = [well["path"] for well in plate_data.get("wells")]
    # well_paths.sort()

    # Get the first well...
    well_group = get_first_well(plate_group)
    first_field_path = get_first_field_path(well_group)

    if labels_path:
        first_field_path = first_field_path + "/labels/" + labels_path

    print("get_pyramid_lazy: first_field_path", first_field_path)
    image_group = well_group[first_field_path]

    # We assume all images are same shape & dtype as the first one
    paths = [ds["path"] for ds in get_attrs(image_group)["multiscales"][0]["datasets"]]
    img_pyramid = [da.from_zarr(image_group[path]) for path in paths]
    img_pyramid_shapes = [d.shape for d in img_pyramid]
    numpy_type = img_pyramid[0].dtype

    # Create a dask pyramid for the plate
    pyramid = []
    for level, tile_shape in enumerate(img_pyramid_shapes):
        lazy_plate = get_stitched_grid(
            plate_group, level, tile_shape, numpy_type, first_field_path
        )
        pyramid.append(lazy_plate)

    # Use the first image's metadata for viewing the whole Plate
    # node.metadata = well_spec.img_metadata

    # "metadata" dict gets added to each 'plate' layer in napari
    # node.metadata.update({"metadata": {"plate": self.plate_data}})
    return pyramid


def get_stitched_grid(
    plate_group, level: int, tile_shape: tuple, numpy_type, first_field_path
) -> da.core.Array:
    plate_data = get_attrs(plate_group)["plate"]
    rows = plate_data.get("rows")
    columns = plate_data.get("columns")
    row_names = [row["name"] for row in rows]
    col_names = [col["name"] for col in columns]

    well_paths = [well["path"] for well in plate_data.get("wells")]
    well_paths.sort()

    row_count = len(rows)
    column_count = len(columns)

    def get_tile(row: int, col: int) -> da.core.Array:
        """tile_name is 'level,z,c,t,row,col'"""

        # check whether the Well exists at this row/column
        well_path = f"{row_names[row]}/{col_names[col]}"
        if well_path not in well_paths:
            return np.zeros(tile_shape, dtype=numpy_type)

        img_path = f"{well_path}/{first_field_path}/{level}"

        try:
            # this is a dask array - data not loaded from source yet
            data = da.from_zarr(plate_group[img_path])
        except ValueError:
            # FIXME: check the Well to get the actual first field path
            data = da.zeros(tile_shape, dtype=numpy_type)
        return data

    lazy_rows = []
    # For level 0, return whole image for each tile
    for row in range(row_count):
        lazy_row: list[da.Array] = [get_tile(row, col) for col in range(column_count)]
        lazy_rows.append(da.concatenate(lazy_row, axis=len(lazy_row[0].shape) - 1))
    return da.concatenate(lazy_rows, axis=len(lazy_rows[0].shape) - 2)


def get_first_well(plate_group):
    plate_data = get_attrs(plate_group)["plate"]
    well_paths = [well["path"] for well in plate_data.get("wells")]
    well_paths.sort()

    # Get the first well...
    well_group = plate_group[well_paths[0]]
    if well_group is None:
        raise Exception("Could not find first well")
    return well_group


def get_first_field_path(well_group):
    well_data = get_attrs(well_group)["well"]
    if well_data is None:
        raise Exception("Could not find well data")

    first_field_path = well_data["images"][0]["path"]
    return first_field_path
