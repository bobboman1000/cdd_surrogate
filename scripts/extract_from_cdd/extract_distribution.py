### Original Code by PSE-CDD project. Some modifications were applied.


#__author__ = 'sudmanns'
#searchfile = open("log", "r")
#for line in searchfile:
#    if "Stress" in line:
#        print line
#searchfile.close()
import re


def extract_distribution(path):
    with open(path + "/log", "r") as input_file_1, \
         open(path + '/distribution_moment.dat', 'w') as output_file_1:

        for line in input_file_1:
            if re.match(r'.*\\M_T\\b', line):
                output_file_1.write(line)


if __name__ == "__main__":
    extract_distribution("")


