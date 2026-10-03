"""Desktop launch and offline command-line tools share the same entry point."""
from .cli import main

if __name__ == '__main__':
    raise SystemExit(main())
