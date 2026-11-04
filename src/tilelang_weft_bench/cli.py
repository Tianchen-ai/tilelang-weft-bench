"""Select the TileLang catalog without replacing the shared CLI or runners."""

def main(argv=None):
    from weft_bench.cli import main as run

    return run(argv, catalog_provider="tilelang_weft_bench.catalogs")
