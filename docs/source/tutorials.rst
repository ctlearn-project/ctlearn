*********
Tutorials
*********

Public CTAO data exploration
============================

This tutorial follows the executable Python-script style used by ctapipe's
tutorials. It downloads a small SimTel sample for the monoscopic examples and
four CTAO simulation files from `Zenodo record 7298569
<https://zenodo.org/records/7298569>`_ for the stereo example. The four Zenodo
files require about 3.5 GB of disk space. Downloaded and converted files are
reused on subsequent runs.

To run the complete tutorial from the CTLearn repository root, install CTLearn
with its development dependencies and execute:

.. code-block:: console

   python examples/tutorials/explore_public_data.py

The source is included below for reading and copying. Building the
documentation does not execute the script or download the data.

.. literalinclude:: ../../examples/tutorials/explore_public_data.py
   :language: python
   :linenos:
