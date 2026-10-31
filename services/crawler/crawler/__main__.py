import argparse
import json
import logging

from .connectors import CONNECTORS
from .run import DEFAULT_CONFIG, crawl

parser = argparse.ArgumentParser(prog="crawler", description="Collect raw documents into HDFS bronze.")
parser.add_argument("--sources", nargs="+", choices=list(CONNECTORS), help="default: all enabled in config")
parser.add_argument("--config", default=DEFAULT_CONFIG)
parser.add_argument("--dry-run", action="store_true", help="fetch and log, do not write to HDFS")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
print(json.dumps(crawl(args.sources, args.config, args.dry_run), indent=2))
