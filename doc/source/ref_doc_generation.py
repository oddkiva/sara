import logging
import os

from jinja2 import Environment, FileSystemLoader


REF_TEMPLATE_DIR = os.path.join('..', 'ref_doc_templates')

DO_SOURCE_DIR = '../../src/DO'

DO_LIBRARIES = [
    'Core',
    'Graphics',
    'FileSystem',
    'Geometry',
    'ImageProcessing',
    'Features',
    'FeatureDescriptors',
    'FeatureDetectors',
    'FeatureMatching',
    'Match'
]

logger = logging.getLogger(__name__)

env = Environment(loader=FileSystemLoader(REF_TEMPLATE_DIR))


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
        'Core': (
            DO_SOURCE_DIR,
            ['Core.hpp', 'Core/Timer.hpp', 'Core/Color.hpp', ...]
        ),
        'Graphics': (
            DO_SOURCE_DIR,
            ['Graphics.hpp', ...]
        ), ...

    """

    breathe_projects_source = {
        library: (
            DO_SOURCE_DIR,
            ['{}.hpp'.format(library)] + list_source_files(library)
        )
        for library in DO_LIBRARIES
    }
    return breathe_projects_source


def generate_ref_doc(library):
    """ This generates automatically rst files for the reference documentation
    of the target library.

    Parameters
    ----------
    library: str
        The library name.

    """

    template = env.get_template('library.rst')

    # Generate the context data.
    title = 'DO::{}'.format(library)
    section_markup = '=' * len(title)
    section = '\n'.join([title, section_markup])

    context_data = {
        'section': section,
        'library': library,
    }

    # Render the template.
    output = template.render(**context_data)

    output_dir = 'reference'
    try:
        os.makedirs(output_dir)
    except:
        pass

    # Save to file.
    output_file_path = "{}.rst".format(os.path.join(output_dir, library))
    with open(output_file_path, 'w') as output_file:
        output_file.write(output)


def generate_all_ref_doc():
    """ Convenience function to generate all the reference documentation.

    """

    for library in DO_LIBRARIES:
        generate_ref_doc(library)


if __name__ == '__main__':
    generate_all_ref_doc()
