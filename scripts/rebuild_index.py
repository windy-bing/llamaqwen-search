from app.config import get_settings
from app.rag import RagService


def main() -> None:
    settings = get_settings()
    settings.rebuild_index = True
    RagService(settings).load()
    print(f"Index rebuilt at {settings.index_dir}")


if __name__ == "__main__":
    main()

