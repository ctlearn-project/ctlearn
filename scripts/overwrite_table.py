import argparse
from ctapipe.io import read_table, write_table


def overwrite_table(to_be_overwritten, overwrite_from, table_path):
    """
    Overwrite a table in a ctapipe hdf5 file with another table

    This function reads a table from a ctapipe hdf5 file 'overwrite_from'
    and overwrites a table in another ctapipe hdf5 file 'to_be_overwritten'
    with it.

    Parameters
    ----------
    to_be_overwritten : str
        ctapipe hdf5 file with the table to be overwritten
    overwrite_from : str
        ctapipe hdf5 file with the table to overwrite from
    table_path : str
        path to the table in the ctapipe hdf5 file to be overwritten
    """
    # Read the table from the 'overwrite_from' file
    table = read_table(overwrite_from, table_path)
    # Overwrite the table in the 'to_be_overwritten' file
    write_table(
        table,
        to_be_overwritten,
        table_path,
        overwrite=True,
    )
    print(
        f"Table at {table_path} in {to_be_overwritten} overwritten with table from {overwrite_from}."
    )


def main():
    parser = argparse.ArgumentParser(
        description=("Overwrite a table in a ctapipe hdf5 file with another table")
    )
    parser.add_argument(
        "--to_be_overwritten", help="ctapipe hdf5 file with the table to be overwritten"
    )
    parser.add_argument(
        "--overwrite_from", help="ctapipe hdf5 file with the table to overwrite from"
    )
    parser.add_argument(
        "--table_path",
        help="path to the table in the ctapipe hdf5 file to be overwritten",
        default="/dl2/event/telescope/geometry/CTLearn/tel_001",
    )
    args = parser.parse_args()

    overwrite_table(args.to_be_overwritten, args.overwrite_from, args.table_path)


if __name__ == "__main__":
    main()
