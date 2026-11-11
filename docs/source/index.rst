.. CTLearn documentation master file, created by
   sphinx-quickstart on Mon Mar 28 13:43:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true

CTLearn: Deep Learning for IACT Event Reconstruction
====================================================

.. image:: _static/ctlearn_logo_transparent.png
   :align: center
   :width: 75%

.. CAUTION:: This is not production-ready code yet. Although stable
   releases are available, users should expect changes to the code
   before a 1.0 release arrives.

.. toctree::
   :maxdepth: 1
   :hidden:

   userguide
   devguide
   api
   Changelog <changelog>

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: User Guide
      :link: userguide
      :link-type: doc

      Explore the public CTAO data tutorials, train CTLearn models, and run
      inference on image and waveform data.

   .. grid-item-card:: Developer Guide
      :link: devguide
      :link-type: doc

      Set up a development environment and contribute to CTLearn.

   .. grid-item-card:: API Docs
      :link: api
      :link-type: doc

      Browse the CTLearn Python package and its modules.
