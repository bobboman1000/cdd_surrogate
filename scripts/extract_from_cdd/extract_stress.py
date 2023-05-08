### Original Code by PSE-CDD project. Some modifications were applied.

#__author__ = 'sudmanns'
#searchfile = open("log", "r")
#for line in searchfile:
#    if "Stress" in line:
#        print line
#searchfile.close()
import re


def extract_stress(path):
    with open(path + "/log", "r") as input_file_1, \
         open(path + '/stress_norm.dat', 'w') as output_file_1, \
         open(path + '/strain_norm.dat', 'w') as output_file_2, \
         open(path + '/strain-stress_yy.dat', 'w') as output_file_3, \
         open(path + '/strain-stress_xy.dat', 'w') as output_file_4:

        for line in input_file_1:
            if re.match(r'.*\b_S\b', line):
                output_file_1.write(line)
            if re.match(r'.*\b_E\b', line):
                output_file_2.write(line)
            if re.match(r'.*\beps_yy\b', line):
                output_file_3.write(line)
            if re.match(r'.*\beps_xy\b', line):
                output_file_4.write(line)

if __name__ == "__main__":
    extract_stress("")