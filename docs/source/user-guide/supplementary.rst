=====================
Supplementary scripts
=====================

The ``scripts/`` directory contains utilities for modifying existing ctapipe HDF5
files:

``append_subarray_table.py``
    Build subarray-level DL2 predictions from the telescope-level predictions in
    an input file. For each configured reconstruction task (classification,
    energy, and geometry by default), the script combines the telescope tables
    with a ctapipe stereo combiner and writes the result back to the same file
    under ``/dl2/event/subarray/<task>/<prefix>``. The default prefix is
    ``CTLearn``.

    Set the input file with the ctapipe tool option:

    .. code-block:: console

       python scripts/append_subarray_table.py --AppendSubarrayTable.input_url=predictions.dl2.h5

``overwrite_metadata.py``
    Migrate an HDF5 file's product metadata to the current CTAO data model.
    The script reads metadata through ctapipe's migration-aware reader and
    writes the migrated metadata back to the same file, removing legacy
    metadata attributes without intentionally changing metadata values.

    .. code-block:: console

       python scripts/overwrite_metadata.py --file_path predictions.dl2.h5

    This requires ctapipe 0.33.0 or newer.

``overwrite_table.py``
    Replace one table in a ctapipe HDF5 file with the corresponding table from
    another file, leaving the rest of the destination file unchanged. Specify
    the destination with ``--to_be_overwritten``, the source with
    ``--overwrite_from``, and the HDF5 table path with ``--table_path``. The
    default table path is
    ``/dl2/event/telescope/geometry/CTLearn/tel_001``.
