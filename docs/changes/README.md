# How to use Towncrier

CTLearn uses [Towncrier](https://towncrier.readthedocs.io/en/stable/) to collect
changelog entries from pull requests.

## Add a changelog fragment

Create a reStructuredText file named `<PULL REQUEST>.<TYPE>.rst` in this
directory. Replace `<PULL REQUEST>` with the pull request number and choose one
of these types:

- `feature`: New Features
- `bugfix`: Bug Fixes
- `api`: API Changes
- `datamodel`: Data Model Changes
- `optimization`: Refactoring and Optimization
- `maintenance`: Maintenance

Write a short description of the user-visible change. For example:

```rst
Fixed ``crazy_function`` to be consistent with ``not_so_crazy_function``.
```

Pull requests that do not need a changelog entry can be exempted by a maintainer
using the `no-changelog-needed` label.

## Build the changelog

Maintainers can generate the release changelog from the repository root with:

```bash
towncrier build --version=<VERSION>
```

This combines the fragments into `CHANGES.rst` and removes the consumed
fragments. Review the generated file before committing it.
