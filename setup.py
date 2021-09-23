from setuptools import setup, find_packages, Command
import os


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


# Load Requirements
with open('requirements.txt') as f:
    requirements = f.readlines()

# Load README
with open('README.md') as readme_file:
    readme = readme_file.read()

setup_requirements = []
test_requirements = []

COMMANDS = []

data_files = []

setup(
    author="drkostas",
    author_email="georgiou.kostas94@gmail.com",
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    cmdclass={
        'clean': CleanCommand,
    },
    data_files=[('', data_files)],
    description="Code for the Machine Learning Course (COSC-522) of the UTK.",
    entry_points={'console_scripts': COMMANDS},
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='cosc522',
    name='cosc522',
    # package_dir={'': '.'},
    packages=find_packages(include=['custom_libs',
                                    'custom_libs.*']),
    # py_modules=['main'],
    setup_requires=setup_requirements,
    url='https://github.com/drkostas/COSC522.git',
    version='0.1.0',
    zip_safe=False,
)
