import sys
from urllib.request import urlopen


def download(url, filename):
    try:
        print(f"Downloading from {url} ...", file=sys.stderr)
        data = urlopen(url).read()
        with open(filename, mode="wb") as f:
            f.write(data)
    except:
        print(f"Failed to downalod the data from {url}", file=sys.stderr)
        sys.exit(1)
