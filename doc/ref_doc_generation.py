import logging
import os

from jinja2 import Environment, FileSystemLoader


# Input source directory and input list of directories.
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DO_PROJECT_NAME = 'DO-CV'
DO_SOURCE_DIR = os.path.join(CURRENT_DIR, '../src/DO')
DO_LIBRARIES = sorted(os.walk(DO_SOURCE_DIR).next()[1])

# Input directory containing Jinja2-based templates.
REF_DOC_TEMPLATE_DIR = 'ref_doc_templates'

# Output directory in which we put the reference documentation.
OUTPUT_REF_DOC_DIR = 'source/reference'

logger = logging.getLogger(__name__)

env = Environment(loader=FileSystemLoader(REF_DOC_TEMPLATE_DIR))


def list_source_files(library):
    """ Establish the list of header files that constitute the target library.

    Parameters
    ----------
    library: str
        The library name.

    """

    # Get the absolute path of the module.
    library_dir_path = os.path.join(DO_SOURCE_DIR, library)

    source_files = []
    for dir, sub_dirs, files in os.walk(library_dir_path):
        logger.info('Exploring directory: {}'.format(dir))

        # Get the relative path of the directory.
        dir_relpath = os.path.relpath(dir, DO_SOURCE_DIR)
        if dir_relpath == '.':
            dir_relpath = ''

        # Get the list of header files
        for file in files:
            if file.endswith('.hpp'):
                file_relpath = os.path.join(dir_relpath, file)
                source_files.append(file_relpath)
                logger.info('Appended file: {}'.format(file_relpath))

    return source_files


def list_projects_source():
    """ Populate the list of projects source for breathe.

    `breathe_projects_source` should be of the following form:
    breathe_projects_source = {
        'DO-CV': (
            DO_SOURCE_DIR,
            ['Core.hpp', 'Core/Timer.hpp', 'Core/Color.hpp', ...
             'Graphics.hpp', ...]
        )
    }

    """

    header_files = []
    for library in DO_LIBRARIES:
        master_header_file = '{}.hpp'.format(library)
        modules = list_source_files(library)
        header_files.append(master_header_file)
        header_files.extend(modules)

    breathe_projects_source = {
        DO_PROJECT_NAME: (DO_SOURCE_DIR, header_files)
    }
    return breathe_projects_source


def generate_section(title):
    markup = '=' * len(title)
    return '\n'.join([title, markup])


def generate_module_doc(library, module):
    """ Generate the reference documentation of the module.

    library: str
        The name of the library.

    module: str
        The name of the module.
    """

    module_dir = os.path.dirname(module)

    # Create the directory if necessary.
    output_dir = os.path.join(OUTPUT_REF_DOC_DIR, module_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the reference documentation.
    template = env.get_template('module.rst')
    title, _ = os.path.splitext(os.path.basename(module))
    context_data = {
        'section': generate_section(title),
        'module': os.path.basename(module),
        'library': DO_PROJECT_NAME
    }
    rendered_template = template.render(**context_data)

    # Save the rendered documentation to file.
    output_file_path = '{}.rst'.format(os.path.join(output_dir, title))
    with open(output_file_path, 'w') as output_file:
        output_file.write(rendered_template)


def generate_ref_doc(library, module_list):
    """ This generates automatically rst files for the reference documentation
    of the target library.

    Parameters
    ----------
    library: str
        The library name.

    module_list: list(str)
        The list of modules that constitutes the library.

    """

    # Render the reference documentation.
    template = env.get_template('library.rst')
    # The list of documentation files for each module.
    doc_files = [os.path.splitext(module)[0] for module in module_list]
    context_data = {
        'section': generate_section(library),
        'library': library,
        'modules': doc_files
    }
    rendered_template = template.render(**context_data)

    # Save the rendered documentation to file.
    if not os.path.exists(OUTPUT_REF_DOC_DIR):
        os.makedirs(OUTPUT_REF_DOC_DIR)

    output_file_path = '{}.rst'.format(os.path.join(OUTPUT_REF_DOC_DIR,
                                                    library))
    with open(output_file_path, 'w') as output_file:
        output_file.write(rendered_template)


def generate_ref_doc_toc():
    """ Generate the table of contents of the reference documentation.

    """

    template = env.get_template('ref_doc_toc.rst')
    context_data = {
        'libraries': DO_LIBRARIES
    }
    rendered_template = template.render(**context_data)
    with open(os.path.join('source', 'ref_doc_toc.rst'), 'w') as output_file:
        output_file.write(rendered_template)


def generate_all_ref_doc():
    """ Convenience function to generate all the reference documentation.

    """

    # Generate the documentation index.
    generate_ref_doc_toc()

    for library in DO_LIBRARIES:
        # Remove the master header file from the list of modules.
        module_list = list_source_files(library)
        # Generate documentation index of the library.
        generate_ref_doc(library, module_list)
        # Generate documentation file for each module of the library.
        for module in module_list:
            generate_module_doc(library, module)


if __name__ == '__main__':
    generate_all_ref_doc()

    sources = list_projects_source()
