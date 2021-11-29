.. _dev:

================
Developers Guide
================

For developers of nuSpaceSim to perform needed improvements, maintenance, and upgrades.

Access Source Code
------------------

The version control source repository for nuspacesim is on github.

::

  git clone https://github.com/NuSpaceSim/nuSpaceSim.git
  cd nuSpaceSim


Prepare Development Environment
-------------------------------

There are a few development tools used by nuspacesim that are not required for users.
To fully develop nuspacesim you should install them in your local python environment with

::

  python3 -m pip install -r requirements-dev.txt

These requirements are:

- black: source code layout formatting
- flake8: source code health checking
- pre-commit: enforce requirements before committing
- pytest: unit test framework
- tox: python version environment manager and unit test system.


Development Build & Install
---------------------------

Use the -e flag on pip to build the nuspacesim python wheel as an editable install. This
method compiles the code locally and copies a symbolic link to the code into your active
python site-packages directory. The consequence of this is that making source code changes
to local python files are instantly updated in the installed executable.

::

  python3 -m pip install -e .



Build wheel package from source
-------------------------------

The python wheel is the compiled binary package which is distributed on pypi.

::

  python3 -m pip wheel -w dist --use-feature=in-tree-build --no-deps  .


Build Documentation
-------------------

The documentation for nuspacesim is compiled from RST files using sphinx. The
required dependencies ar found in `docs/requirements.txt` and should be installed in
your development environment with

::

  pip install -r docs/requirements.txt


Build Documentation with tox
----------------------------

::

  python3 -m pip install tox
  python3 -m tox -e docs


Run unit tests
--------------
::

  python3 -m pip install pytest
  python3 -m pytest test


Run unit tests on multiple versions of python
---------------------------------------------
::

  tox .
