import pandas as pd


def read_csv(file_path: str, column_name: str, sep: str | None = None, **kwargs) -> list[str]:
    df = pd.read_csv(file_path, sep=sep, **kwargs)
    return df[column_name].to_list()


def _strip(line: str) -> str:
    return line.replace('\n', '').replace('"', '').strip()


def read_txt(file_path: str, skip_header: bool) -> list[str]:
    with open(file_path, "r") as f:
        lines = f.readlines()

    if skip_header:
        lines = lines[1:]

    lines = map(_strip, lines)
    lines = filter(lambda line: len(line) > 0, lines)
    return list(lines)
