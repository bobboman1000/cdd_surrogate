### Original Code by PSE-CDD project. Some modifications were applied.


#__author__ = 'sudmanns'
#searchfile = open("log", "r")
#for line in searchfile:
#    if "Stress" in line:
#        print line
#searchfile.close()
import re

def extract_dislocation(path):
    with open(path + "/log", "r") as input_file_1, \
         open(path + '/dislocation_sum.dat', 'w') as output_file_1, \
         open(path + "/dislocation_ns.dat", 'w') as output_file_2:

        for line in input_file_1:
            if re.match(r'.*\bsum_n_prod\b', line):
                output_file_1.write(line)
            if re.match(r'.*\bsum_n_prod_ns\b', line):
                output_file_2.write(line)


if __name__ == "__main__":
    extract_dislocation("")