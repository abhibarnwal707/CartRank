import sys

from cartrank.data import esci, reviews

COMMANDS = {
    "esci-splits": esci.save_splits,
    "reviews": reviews.save_all,
}

if __name__ == "__main__":
    COMMANDS[sys.argv[1]]()
