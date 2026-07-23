from typing import Any
import numpy as np
from ome_zarr import OMEZarrImage, OMEZarrMultiscale
from napari.qt.threading import thread_worker
from napari.utils import progress
from napari.qt import create_worker
import time

def write_05_ome_zarr(path: str, layer_data, attributes: dict) -> list[str]:

    axes = "tzyx"

    # check if first color is black - this indicates a non-inverted colormap
    first_color = attributes["colormap"]["colors"][0]
    if all([i == 0 for i in first_color[:-1]]):
        rgba = np.asarray(attributes["colormap"]["colors"][-1])
    else:
        rgba = np.asarray(first_color)

    if not attributes.get("multiscale", False):
        axes = axes[-len(layer_data.shape) :]
        img = OMEZarrImage(
            data=layer_data,
            axes=axes,
            scale={d: float(attributes["scale"][i]) for i, d in enumerate(axes)},
            name=attributes.get("name", "image"),
        )

        img_ms = OMEZarrMultiscale(
            image=img,
            contrast_limits=[attributes.get("contrast_limits", None)],
            channel_colors=[[int(c*255) for c in rgba]]
        )

        jobs = img_ms.to_ome_zarr(path, overwrite=True, compute=False)

        @thread_worker(progress={"total": len(jobs)})
        def save_pyramid_layers(jobs):
            for job in jobs:
                job.compute()
                yield

        save_pyramid_layers(jobs).start()

    # return path to any file(s) that were successfully written
    return [path]