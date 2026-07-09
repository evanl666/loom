"""Enable ``python -m loom`` as an alias for the ``loom`` CLI."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
