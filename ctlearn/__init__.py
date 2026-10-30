from importlib.metadata import metadata

from ._version import __version__


def get_repository_url() -> str:
    for project_url in metadata("ctlearn").get_all("Project-URL", []):
        name, separator, url = project_url.partition(",")
        if separator and name.strip().casefold() == "repository":
            return url.strip()
    raise LookupError("The ctlearn project metadata does not define a repository URL")


__all__ = ["__version__", "get_repository_url"]
