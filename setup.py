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

# For the cases you want a different package to be installed on local and prod environments
# import subprocess
# LOCAL_ARG = '--local'
# if LOCAL_ARG in sys.argv:
#     index = sys.argv.index(LOCAL_ARG)  # Index of the local argument
#     sys.argv.pop(index)  # Removes the local argument in order to prevent the setup() error
#     subprocess.check_call([sys.executable, "-m", "pip", "install", 'A package that works locally'])
# else:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", 'A package that works on production'])

# Load README
with open('README.md') as readme_file:
    readme = readme_file.read()

setup_requirements = []
test_requirements = []

COMMANDS = [
    'cli = cosc522.cli:app',
    'cosc522_main = cosc522.main:main'
]

data_files = ['cosc522/configuration/yml_schema.json']

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
    packages=find_packages(include=['cosc522',
                                    'cosc522.*']),
    # py_modules=['main'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/drkostas/COSC522.git',
    version='0.1.0',
    zip_safe=False,
)
