'''
Setup for my_package that describes it's contents and how to install it.
'''
from setuptools import setup

# List of dependencies for the package
# >= Can be used to specify a minimum version
# >=,< Can be used to specify a minimum and maximum version e.g. 'numpy>=1.15,<1.20',
dependencies = [
    'argparse>=1.4.0',
    'numpy>=1.15',
    'matplotlib>=2.1.0',
    'astropy>=2.0.2',
    'pandas',
]

setup(
    # Name of the repository
    name='my_project',
    # Version of the software (keep this up to date with git tags/releases)
    version=1.0,
    # Short description of the package
    description='An example package to be used as a template',
    # Update this with your github URL
    url='https://github.com/ADACS-Australia/python_project_template',
    # Minimum version of python required
    python_requires='>=3.6',
    # Name of your package, should be the same as the name of the directory
    packages=['my_package', 'my_package.scripts'],
    # Any data files that need to be included with the package (non python files)
    package_data={'my_package':['data/*.csv']},
    # Dependencies for the package
    install_requires=dependencies,
    # Scripts that will be run on the command line
    entry_points={
        'console_scripts': [
            # Make a hello_world command that runs the main function in hello_world.py script located in my_package/scripts
            'hello_world=my_package.scripts.hello_world:main',
            'filter_and_plot=my_package.scripts.filter_and_plot:main',
        ],
    },
)