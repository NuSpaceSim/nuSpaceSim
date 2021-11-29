.. _nuspacesim_docs_mainpage:

.. image:: _static/NuSpaceSimLogoBlack.png
   :alt: NuSpaceSim Logo Black

========================
nuSpaceSim Documentation
========================

νSpaceSim

This is the beta release of the nuspacesim simulator tool!

This package simulates upward-going electromagnetic air showers caused by neutrino
interactions with the atmosphere. It calculates the tau neutrino acceptance for the
Optical Cherenkov technique. The simulation is parameterized by an input XML
configuration file, with settings for detector characteristics and global parameters.
The package also provides a python3 API for programatic access.

Tau propagation is interpolated using included data tables from nuPyProp.

.. panels::
    :container: container-fluid pb-1
    :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-2
    :card: + intro-card text-center
    :img-top-cls: pl-5 pr-5

    ---
    :opticon:`rocket` Quickstart
    ^^^^^^^^^^^^^^^
    New to νSpaceSim? Check out the Quickstart guide for help installing and
    navigating the package.
    +++

    .. link-button:: quickstart
            :type: ref
            :text: To the quickstart guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :fa:`book` User guide
    ^^^^^^^^^^
    The user guide provides in-depth information on the
    key concepts of nuSpaceSim with useful background information and explanation.
    +++

    .. link-button:: tutorial
            :type: ref
            :text: To the user guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :fa:`cogs` API reference
    ^^^^^^^^^^^^^
    The reference guide contains a detailed description of the how the methods work
    and which parameters can be used.
    +++

    .. link-button:: reference
            :type: ref
            :text: To the reference guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :fa:`code` Developer guide
    ^^^^^^^^^^^^^^^
    Need to modify or build nuSpaceSim from source? Understand the testing pipeline?
    Developer guides are here for you!
    +++

    .. link-button:: dev
            :type: ref
            :text: To the developer guide
            :classes: btn-block btn-secondary stretched-link

.. toctree::
   :hidden:

   Quickstart <quickstart>
   User Guide <tutorial/index>
   API reference <reference/index>
   Development <dev/index>

