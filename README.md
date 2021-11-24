# COSC 522

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/drkostas/cosc522/master/LICENSE)

## Table of Contents

+ [About](#about)
+ [TODO](#todo)
+ [Libraries Overview](#lib_overview) 
+ [Prerequisites](#prerequisites)
+ [Bootstrap Project](#bootstrap)
+ [Running the code using Jupyter](#jupyter)
    + [Local Jupyter](#local_jupyter)
    + [Google Collab](#google_collab)
+ [Adding New Libraries](#adding_libs) 
+ [License](#license)

## About <a name = "about"></a>

Code for the Machine Learning Course (COSC-522) of the UTK.

## TODO <a name = "todo"></a>

Read the [TODO](TODO.md) to see the current task list.

## Libraries Overview <a name = "lib_overview"></a>

All the libraries are located under *"\<project root>/custom_libs"*
- ***Project1***: Loc: **Project1**, Desc: *Code needed in Project 1*
- ***Project2***: Loc: **Project2**, Desc: *Code needed in Project 2*
- ***Project3***: Loc: **Project3**, Desc: *Code needed in Project 3*
- ***Project4***: Loc: **Project4**, Desc: *Code needed in Project 4*
- ***Project5***: Loc: **Project5**, Desc: *Code needed in Project 5*
- ***CARLO***: Loc: **CARLO**, Desc: *RL/Pygame code used for Project 5*
- ***ColorizedLogger***: Loc: **fancy_logger/colorized_logger.py**, Desc: *Logger with formatted text capabilities*
- ***timeit***: Loc: **timing_tools/timeit.py**, Desc: *Decorator/ContextManager for counting the execution times of functions and code blocks*

## Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python >= 3.7 and any Bash based shell (e.g. zsh) installed.
Having installed conda is also recommended.

```Shell

$ python3.7 -V
Python 3.7.12

$ echo $SHELL
/usr/bin/zsh

```

## Bootstrap Project <a name = "bootstrap"></a>

**This is only needed if you're running the code locally and NOT in Google Collab.**

All the installation steps are being handled by the [Makefile](Makefile).

If you want to use conda run:
```Shell
$ make install
```

If you want to use venv run:
```Shell
$ make install env=venv
```

## Using Jupyter <a name = "jupyter"></a>

In order to run the code, you will only need to configure the yml file, and either run its
file directly or invoke its console script. Refer to [Configuration](#configuration) Section.

### Local Jupyter <a name = "local_jupyter"></a>

First, make sure you are in the correct virtual environment:

```Shell
$ conda activate cosc522

$ which python
/home/<your user>/anaconda3/envs/cosc522/bin/python
```

To use jupyter, first run `jupyter`:

```shell
jupyter notebook
```
And open the [main.ipynb](main.ipynb).

### Google Collab <a name = "google_collab"></a>

Just Open this [Google Collab Link](https://colab.research.google.com/drive/1evpodmjkOM1_NzyinYWJCz4xVRHAXZb6).

## Adding New Libraries <a name = "adding_libs"></a>

If you want to add a new library (e.g. a Class) in the project you need to follow these steps:
1. Create a new folder under *"\<project root>/custom_libs"* with a name like *my_lib*
2. Create a new python file inside it with a name like *my_module.py*
3. Paste your code inside it
4. Create a new file name *__init__.py*
5. Paste the follwing code inside it:
   ```python
    """<Library name> sub-package."""
    
    from .<Module name> import <Class Name>
    
    __email__ = "georgiou.kostas94@gmail.com"
    __author__ = "drkostas"
    __version__ = "0.1.0"
    ```
6. Open [\<project root>/custom_libs/\_\_init\_\_.py](custom_libs/__init__.py)
7. Add the following line: ```from custom_libs.<Module name> import <Class Name>```
8. (Optional) Rerun `make install` or `python setup.py install` 
 
## License <a name = "license"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


