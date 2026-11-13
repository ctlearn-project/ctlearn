.. _dev-guide:

***************
Developer Guide
***************

.. currentmodule:: ctlearn

Getting Started for Developers
------------------------------

As a developer, clone the repository and install CTLearn in development mode.
We support two methods: *pip/venv* and `Pixi <https://pixi.sh/>`__.

.. tab-set::

    .. tab-item:: With Pixi

        Install Pixi if needed, then clone the repository and install the
        development environment:

        .. code-block:: shell

            $ git clone https://github.com/ctlearn-project/ctlearn.git
            $ cd ctlearn
            $ pixi install -e dev

        Useful commands from the repository directory:

        .. code-block:: shell

            $ pixi run -e dev test            # run the test suite
            $ pixi run -e test-py313 test     # run tests with Python 3.13
            $ pixi run -e test-py314 test     # run tests with Python 3.14
            $ pixi run -e dev make -C docs html
            $ pixi shell -e dev               # open a shell in the dev environment
            $ pixi clean                      # remove Pixi environments

        Pixi targets Linux x86_64 and Apple Silicon macOS.

    .. tab-item:: With Pip

        .. code-block:: shell

            $ git clone https://github.com/ctlearn-project/ctlearn.git
            $ cd ctlearn
            $ python -m venv .venv
            $ source .venv/bin/activate
            $ python -m pip install --upgrade pip
            $ python -m pip install -e ".[dev]"

Contributing
------------

.. toctree::
   :maxdepth: 1

   contributing



Core Dependencies
-----------------

* python>=3.12
* torch>=2.4.0
* tensorflow>=2.18
* keras>=3.0
* astropy
* ctapipe>=0.29.0
* dl1_data_handler>=0.14.13
* numba
* numpy
* pandas
* pyyaml
