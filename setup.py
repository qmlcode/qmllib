import sys
from pathlib import Path

from setuptools import setup

try:
    import _compile
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    import _compile

if __name__ == "__main__":
    _compile.main()
    setup()
