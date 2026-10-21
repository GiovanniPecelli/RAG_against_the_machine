import fire

try:
    from .rag_against_the_machine.__main__ import RAGCLI
except ImportError:
    from rag_against_the_machine.__main__ import RAGCLI


def main() -> None:
    """Entry point for the CLI."""
    fire.Fire(RAGCLI())


if __name__ == "__main__":
    main()
