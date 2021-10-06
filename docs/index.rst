.. _nuspacesim_docs_mainpage:

.. module:: nuspacesim

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
    :card: + intro-card text-center

    ---
    Getting started
    ^^^^^^^^^^^^^^^

    New to νSpaceSim? Check out the Quickstart guide for help installing and
    navigating the package.
    +++

    .. link-button:: quickstart
            :type: ref
            :text: To the quickstart guide
            :classes: btn-block btn-secondary stretched-link

    ---
    .. :img-top: _static/index_user_guide.png

    User guide
    ^^^^^^^^^^

    The user guide provides in-depth information on the
    key concepts of nuSpaceSim with useful background information and explanation.

    +++

    .. link-button:: tutorial
            :type: ref
            :text: To the user guide
            :classes: btn-block btn-secondary stretched-link

    ---
    .. :img-top: _static/index_api.png

    API reference
    ^^^^^^^^^^^^^

    The reference guide contains a detailed description of
    the nss API. The reference describes how the methods work and which parameters can
    be used. It assumes that you have an understanding of the key concepts.

    +++

    .. link-button:: reference
            :type: ref
            :text: To the reference guide
            :classes: btn-block btn-secondary stretched-link

    ---
    .. :img-top: _static/index_contribute.png

    Developer guide
    ^^^^^^^^^^^^^^^

    Saw a typo in the documentation? Want to improve
    existing functionalities? The contributing guidelines will guide
    you through the process of improving nss.

    +++

    .. link-button:: dev
            :type: ref
            :text: To the development guide
            :classes: btn-block btn-secondary stretched-link

.. toctree::
   :maxdepth: 3
   :hidden:

   Getting started <quickstart>
   User Guide <tutorial/index>
   API reference <reference/index>
   Development <dev/index>
