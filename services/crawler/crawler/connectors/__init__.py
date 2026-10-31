from . import arxiv, local_files, rss, wikipedia

CONNECTORS = {
    "arxiv": arxiv.collect,
    "wikipedia": wikipedia.collect,
    "rss": rss.collect,
    "local": local_files.collect,
}
