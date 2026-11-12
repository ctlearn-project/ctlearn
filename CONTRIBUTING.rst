
Contributing to CTLearn
=======================

CTLearn is open source and welcomes contributions from the community!

Contributing guidelines
-----------------------

In CTLearn we follow the "fork-and-pull" Git workflow:


* Fork the repository on GitHub
* Clone the project to your own machine
* Commit your changes to your own branch
* Push your work back up to your fork on GitHub
* Submit a Pull Request so the core developer team can review your changes

Code quality
------------

CTLearn uses pre-commit hooks for whitespace and file checks, spelling, and
Ruff linting and formatting. Install the development dependencies and set up
the hooks in your checkout with:

.. code-block:: console

   pip install -e ".[dev]"
   pre-commit install

The hooks will run automatically before each commit. To run them manually on
all files, use:

.. code-block:: console

   pre-commit run --all-files

Documentation builds
--------------------

For documentation-only pull requests, apply the ``documentation-only`` label
to skip the pip and Pixi test jobs and the dependent SonarCloud scan. The
documentation build still runs, as do the lint and data-cache jobs and any
other applicable workflow checks. Removing the label restores the skipped
pull request jobs. The label does not affect push builds, including pushes to
``main`` and tags, or the release workflow.

To speed up a pull request's documentation build, apply the
``skip-docs-tutorials`` label. The documentation and tutorial pages are still
built, but the tutorials are not executed, so their outputs are omitted.
Removing the label restores tutorial execution. Push builds always execute the
tutorials, regardless of pull request labels.

Changelog entries
-----------------

Every pull request should include a Towncrier fragment describing its user-facing
changes. See ``docs/changes/README.md`` for the naming conventions and supported
categories. If a change does not warrant a changelog entry, maintainers can apply
the ``no-changelog-needed`` pull request label.

Issues
------

Feel free to submit issues and enhancement requests through our `issues page on GitHub <https://github.com/ctlearn-project/ctlearn/issues>`_.

Copyright and License
---------------------

CTLearn is licensed under the `BSD 3-Clause License <https://opensource.org/licenses/BSD-3-Clause>`_.

CTLearn does require that you make your contributions available under the BSD 3-Clause License in order for it be included in the main repository.
