import argparse
from pathlib import Path

import tables

try:
    from ctapipe.io.metadata import read_ctao_metadata, write_product_metadata_hdf5
except ImportError as error:
    read_ctao_metadata = None
    write_product_metadata_hdf5 = None
    _METADATA_IMPORT_ERROR = error
else:
    _METADATA_IMPORT_ERROR = None


class MetadataMigrationUnavailableError(RuntimeError):
    """Raised when ctapipe does not provide the metadata migration helpers."""


def overwrite_metadata(file_path: Path) -> None:
    """Migrate product metadata in an HDF5 file to the current CTAO model."""
    if read_ctao_metadata is None or write_product_metadata_hdf5 is None:
        raise MetadataMigrationUnavailableError(
            "Cannot migrate metadata because the required ctapipe metadata helpers "
            "could not be imported. Upgrade ctapipe to version 0.33.0 or newer."
        ) from _METADATA_IMPORT_ERROR

    product = read_ctao_metadata(file_path)
    with tables.open_file(file_path, mode="r+") as h5_file:
        write_product_metadata_hdf5(
            product=product,
            h5file=h5_file,
            remove_legacy=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate HDF5 product metadata to the current CTAO model."
    )
    parser.add_argument(
        "--file_path",
        type=Path,
        help="HDF5 file whose product metadata should be migrated",
    )
    args = parser.parse_args()

    try:
        overwrite_metadata(args.file_path)
    except MetadataMigrationUnavailableError as error:
        parser.error(str(error))
    print(f"Migrated product metadata in {args.file_path}")


if __name__ == "__main__":
    main()
