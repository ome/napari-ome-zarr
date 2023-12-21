# napari-ome-zarr

[![License](https://img.shields.io/pypi/l/napari-ome-zarr.svg?color=green)](https://github.com/ome/napari-ome-zarr/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-ome-zarr.svg?color=green)](https://pypi.org/project/napari-ome-zarr)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-ome-zarr.svg?color=green)](https://python.org)
[![tests](https://github.com/ome/napari-ome-zarr/workflows/tests/badge.svg)](https://github.com/ome/napari-ome-zarr/actions)
[![codecov](https://codecov.io/gh/ome/napari-ome-zarr/branch/master/graph/badge.svg)](https://codecov.io/gh/ome/napari-ome-zarr)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ome/napari-ome-zarr/main.svg)](https://results.pre-commit.ci/latest/github/ome/napari-ome-zarr/main)


A reader for zarr backed [OME-NGFF](https://ngff.openmicroscopy.org/) images.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

[Install napari] if not already installed.

You can install `napari-ome-zarr` via [pip]. Activate the same environment as you installed napari into, then:

    pip install napari-ome-zarr

## Usage

Napari will use `napari-ome-zarr` plugin to open images that the plugin recognises as ome-zarr.
The image metadata from OMERO will be used to set channel names and rendering settings
in napari::

    napari "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr/"


If a dialog in napari pops up, encouraging you to choose a reader, choose ``napari-ome-zarr`` and click OK. You can stop it happening with addition of ``--plugin napari-ome-zarr`` as in the example below.

To open a local file::

    napari --plugin napari-ome-zarr 13457227.zarr

OR in python::

    import napari

    viewer = napari.Viewer()
    viewer.open("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0101A/13457537.zarr", plugin="napari-ome-zarr")

    napari.run()


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-ome-zarr" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[Install napari]: https://napari.org/stable/tutorials/fundamentals/installation.html
[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/ome/napari-ome-zarr/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
