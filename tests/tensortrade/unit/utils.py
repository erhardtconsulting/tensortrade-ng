import os


def get_path(path: str) -> str:
    script_path = os.path.dirname(os.path.abspath(__file__))
    return str(os.path.join(script_path, path))