import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--domain-path", type=str, default=None)
args = parser.parse_args()
DOMAIN_PATH = args.domain_path


