import argparse
import logging
import os

from jinja2 import Environment, FileSystemLoader


# Input source directory and input list of directories.
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DO_PROJECT_NAME = 'DO-CV'
DO_SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../cpp/src'))
DO_LIBRARIES = sorted(
    next(os.walk(os.path.join(DO_SOURCE_DIR, 'DO')))[1]
)

# Input directory containing Jinja2-based templates.
REF_DOC_TEMPLATE_DIR = 'ref_doc_templates'

# Output directory in which we put the reference documentation.
OUTPUT_REF_DOC_DIR = 'source/reference'

LOGGER = logging.getLogger(__name__)

ENV = Environment(loader=FileSystemLoader(REF_DOC_TEMPLATE_DIR))


def list_source_files(library_path):
    """ Establish the list of header files that constitute the target library.

    Parameters
    ----------
    library_path: str
        The path of the library directory.

    """

    LOGGER.info("Listing header files in directory: '{}'".format(library_path))

    source_files = []
    for dir, sub_dirs, files in os.walk(library_path):
        # Get the list of header files.
        for file in files:
            if file.endswith('.hpp'):
                file_path = os.path.join(dir, file)
                source_files.append(file_path)
                LOGGER.info("Appended '{}'".format(file_path))

    return sorted(source_files)


def generate_section(title):
    markup = '=' * len(title)
    return '\n'.join([title, markup])


def generate_module_doc(library, module):
    """ Generate the reference documentation of the module.

    library: str
        The name of the library.

    module: str
        The module path.
    """

    module_dirpath = os.path.dirname(module)
    module_relpath = os.path.relpath(module, DO_SOURCE_DIR)
    module_dir_relpath = os.path.relpath(module_dirpath,
                                         os.path.join(DO_SOURCE_DIR, 'DO'))

    # Create the directory if necessary.
    output_dir = os.path.join(OUTPUT_REF_DOC_DIR, module_dir_relpath)
    LOGGER.info("Try to create directory: '{}'".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the reference documentation.
    template = ENV.get_template('module.rst')
    title, _ = os.path.splitext(os.path.basename(module))
    context_data = {
        'section': generate_section(title),
        'module': module_relpath
    }
    LOGGER.info("Generate context data {}".format(context_data))

    rendered_template = template.render(**context_data)

    # Save the rendered documentation to file.
    output_file_path = '{}.rst'.format(os.path.join(output_dir, title))
    with open(output_file_path, 'w') as output_file:
        LOGGER.info("Save module documentation '{}'".format(output_file_path))
        output_file.write(rendered_template)


def generate_library_ref_doc(library, header_filepath_list):
    """ This generates automatically rst files for the reference documentation
    of the target library.

    Parameters
    ----------
    library: str
        The library name.

    header_filepath_list: list(str)
        The list of relative paths of header files in the library.
    """

    # Render the reference documentation.
    template = ENV.get_template('library.rst')
    # The list of documentation files for each module.
    header_filepath_list = [
        os.path.relpath(header, os.path.join(DO_SOURCE_DIR, 'DO'))
        for header in header_filepath_list
    ]
    doc_files = [
        os.path.splitext(header)[0]
        for header in header_filepath_list
    ]
    master_header_filepath = os.path.join('DO', library) + '.hpp'

    context_data = {
        'section': generate_section(library),
        'master_header': master_header_filepath,
        'modules': doc_files
    }
    LOGGER.info("Generate context data {}".format(context_data))

    rendered_template = template.render(**context_data)

    # Save the rendered documentation to file.
    if not os.path.exists(OUTPUT_REF_DOC_DIR):
        os.makedirs(OUTPUT_REF_DOC_DIR)

    output_file_path = '{}.rst'.format(os.path.join(OUTPUT_REF_DOC_DIR,
                                                    library))
    LOGGER.info("Save library doc: '{}'".format(output_file_path))
    with open(output_file_path, 'w') as output_file:
        output_file.write(rendered_template)


def generate_ref_doc_toc():
    """ Generate the table of contents of the reference documentation.
    """

    template = ENV.get_template('ref_doc_toc.rst')
    context_data = {
        'libraries': DO_LIBRARIES
    }
    msg = "Generate table of contents from the following libraries: {}"
    LOGGER.info(msg.format(DO_LIBRARIES))

    rendered_template = template.render(**context_data)
    output_filepath = os.path.join('source', 'ref_doc_toc.rst')
    LOGGER.info("Save ref doc toc: '{}'".format(output_filepath))
    with open(output_filepath, 'w') as output_file:
        output_file.write(rendered_template)


def generate_all_ref_doc():
    """ Convenience function to generate all the reference documentation.
    """

    # Generate the documentation index.
    generate_ref_doc_toc()

    for library in DO_LIBRARIES:
        # Get the list of modules that constitutes the library.
        library_path = os.path.join(DO_SOURCE_DIR, 'DO', library)
        library_module_list = list_source_files(library_path)
        # Generate documentation index of the library.
        generate_library_ref_doc(library, library_module_list)
        # Generate documentation file for each module of the library.
        for module in library_module_list:
            generate_module_doc(library, module)


def parse_command_line_args():
    parser = argparse.ArgumentParser(
        description='Generate the reference documentation'
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="enable verbose logging"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line_args()
    if args.verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.WARNING
    logging.basicConfig(filename='ref_doc_generation.log', level=logging_level)

    generate_all_ref_doc()

    logging.shutdown()
