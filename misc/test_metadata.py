import tables
import numpy as np

from ctalearn.image import MAPPING_TABLES, IMAGE_SHAPES

from ctalearn.data import load_metadata_HDF5

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Print/write metadata for a given collection of standard CTA ML data files."))
    parser.add_argument(
        'file_list',
        help='List of HDF5 (pytables) files to compute metadata for.')
    parser.add_argument(
        '--telescope_type',
        help='Optional telescope type to examine when computing the image-wise class balance. If not provided,
        will compute the event-wise class balance.')
    parser.add_argument(
        '--output_file',
        help='Optional output file to write results to.')
    
    args = parser.parse_args()

    file_list = []
    with open("/home/shevek/datasets/sample_prototype/file_list.txt") as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                file_list.append(line)

    metadata = load_metadata_HDF5(file_list)

    # Get number of examples by file
    num_examples_by_file = metadata['num_images_by_file'][args.telescope_type] if args.telescope_type else metadata['num_events_by_file']

    # Log general information on dataset based on metadata dictionary
    logger.info("%d data files read.", len(file_list))
    logger.info("Telescopes in data:")
    for tel_type in metadata['telescope_ids']:
        logger.info(tel_type + ": "+'[%s]' % ', '.join(map(str,metadata['telescope_ids'][tel_type]))) 
    
    num_examples_by_label = {}
    for i,num_examples in enumerate(num_examples_by_file):
        particle_id = metadata['particle_id_by_file'][i]
        if particle_id not in num_examples_by_label: num_examples_by_label[particle_id] = 0
        num_examples_by_label[particle_id] += num_examples

    total_num_examples = sum(num_examples_by_label.values())

    logger.info("%d total examples.", total_num_examples)
    logger.info("Num examples by label:")
    for label in num_examples_by_label:
        logger.info("%s: %d (%f%%)", label, num_examples_by_label[label], 100 * float(num_examples_by_label[label])/total_num_examples)



    if args.output_file:

    else:
        print(metadata)
