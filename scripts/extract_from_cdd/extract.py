import sys
from extract_dislocation import extract_dislocation
from extract_stress import extract_stress
from extract_distribution import extract_distribution
from extract_params import extract_params


def main(argv):
    assert len(argv) == 1, f"Please provide only 1 arg (path to experiment), {len(argv)} were provided"
    path = argv[0]

    extract_dislocation(path)
    extract_stress(path)
    extract_distribution(path)
    extract_params(path)


if __name__ == "__main__":
    main(sys.argv[1:])
