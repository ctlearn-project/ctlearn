from version import get_version_pypi  

def get_version():
    return get_version_pypi()


if __name__ == "__main__":
    print(get_version())
