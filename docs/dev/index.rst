.. _dev:

================
Developers Guide
================

Download Source Code
--------------------

::

  git clone https://github.com/NuSpaceSim/nuSpaceSim.git
  cd nuSpaceSim


Build and Install Editable
--------------------------

::

  python3 -m pip install -e .



Build wheel package from source
-------------------------------

::

  python3 -m pip wheel -w dist --use-feature=in-tree-build --no-deps  .


Build Documentation
-------------------

::

  python3 -m pip install tox
  python3 -m tox -e docs


Run unit tests
--------------
::

  python3 -m pip install pytest
  python3 -m pytest test
