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

To set up the `pre-commit <https://pre-commit.com/>`_ checks you need to install the
pre-commit hooks into your git repository, this can be done easily with

::

  pre-commit install


Now every attempt to commit updates to the git repository will run the pre-commit
checks, and block any commits that fail.


Development Build & Install
---------------------------

Use the -e flag on `pip` to build the nuspacesim python wheel as an editable install.
This method compiles the code locally and copies a symbolic link to the code into your
active python site-packages directory. The consequence of this is that making source
code changes to local python files are instantly updated in the installed executable.

::

  python3 -m pip install -e .

Now you are able to execute the `nuspacesim` command line application as well as import
the `nuspacesim` package in a python interpreter or script.


Build wheel package from source (Optional)
------------------------------------------

The python wheel is the compiled binary package which is distributed on pypi. You may
want to compile the wheel independently of a pip install because you are testing the
pypi upload. This is an optional step as it is specifically handled by the CICD github
action and does not require developer intervention.

.. code:: sh

  python3 -m pip wheel -w dist --use-feature=in-tree-build --no-deps  .


Build Documentation
-------------------

The documentation for nuspacesim is compiled from RST files using sphinx. This is
a standard best-practice for python packages and open source projects.

Here are links to documentation for the
`RST syntax <https://docutils.sourceforge.io/rst.html>`_ and
`Sphinx <https://www.sphinx-doc.org/en/master/>`_. They describe RST features like
math directives,

.. math::

  α_t(i) = P(O_1, O_2, … O_t, q_t = S_i λ)

  \ln{x} + C = \int{\frac{1}{x}}

Inline math like "The area of a circle is
:math:`A_\text{c} = \left(\frac{\pi d^2}{4}\right)`".
And code blocks like

.. code:: python

   if __name__ == "__main__":
       print("Hello World!")

The documentaion dependencies are found in `docs/requirements.txt` and should be installed
in your development environment with

.. code:: sh

  pip install -r docs/requirements.txt

This does not build the docs themselves, only installs the sphinx dependencies.


Build Documentation with tox
----------------------------

The sphinx-build command is fairly complex, so we've simplified building the
documentation with `tox`, which will compile the code into a new environment then call
the proper sphinx-build command to create the HTML output.

.. code:: sh

  python3 -m tox -e docs

Once this step completes successfully you should be able to open your compiled
documentation in any browser by opening the file path stated at the end of the tox
output. Something like

.. code:: sh

  firefox file:///home/areustle/nasa/NuSpaceSim/nuSpaceSim/.tox/docs_out/index.html


Run unit tests
--------------

The pre-defined unit tests can be run on your system directly with pytest

::

  python3 -m pytest test



Run unit tests on multiple versions of python (Advanced)
--------------------------------------------------------



::

  tox --parallel
