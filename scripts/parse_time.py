import os, sys
import re
import pandas as pd

def ls_absolute(path):
    return [os.path.join(path, filename) for filename in os.listdir(path)]


import os
# https://www.google.com/search?client=safari&rls=en&q=tail+of+file+python&ie=UTF-8&oe=UTF-8
def get_last_n_lines(file_name, N):
    # Create an empty list to keep the track of last N lines
    list_of_lines = []
    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        # Move the cursor to the end of the file
        read_obj.seek(0, os.SEEK_END)
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Get the current position of pointer i.e eof
        pointer_location = read_obj.tell()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            read_obj.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location -1
            # read that byte / character
            new_byte = read_obj.read(1)
            # If the read byte is new line character then it means one line is read
            if new_byte == b'\n':
                # Save the line in list of lines
                list_of_lines.append(buffer.decode()[::-1])
                # If the size of list reaches N, then return the reversed list
                if len(list_of_lines) == N:
                    return list(reversed(list_of_lines))
                # Reinitialize the byte array to save next line
                buffer = bytearray()
            else:
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)
        # As file is read completely, if there is still data in buffer, then its first line.
        if len(buffer) > 0:
            list_of_lines.append(buffer.decode()[::-1])
    # return the reversed list
    return list(reversed(list_of_lines))



def parse_time(path, filename):
    
    file = os.path.join(path, filename)
    
    last_line = None
    for line in get_last_n_lines(file, 10):
        if re.match(r"end program after .* on", line):
            last_line = line
    
    if last_line is None:
        print(path)
        raise Exception("No timestamp found")        
    
    
    # Last line looks like: end program after hh:mm:00.00 minutes on 8 procs at Fri Oct 14 10:31:40 2022
    tokens = last_line.split(" ")
    timestamp = tokens[3]
    proc = tokens[6]
    
    timestamp_tokens = timestamp.split(":")
    if len(timestamp_tokens) == 3:
        hours = int(timestamp_tokens[0])
        minutes = int(timestamp_tokens[1])
        seconds = float(timestamp_tokens[2])
    elif len(timestamp_tokens) == 2:
        hours = 0
        minutes = int(timestamp_tokens[0])
        seconds = float(timestamp_tokens[1])
    else:
        raise Exception(f"No date in appropriate format found in file {path}")
    
    duration_minutes = hours * 60 + minutes + seconds / 60
    
    # Scale time if less than 8 processes
    duration_minutes *= 8 / int(proc)
    
    return duration_minutes


def main(args):    
    target_path = args[0]
    
    
    values = {}
    for dir in ls_absolute(target_path):
        times = [parse_time(dir, "log") for dir in ls_absolute(dir)]
        
        filename = os.path.split(dir)[1]
        values[filename] = times
    
    
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in values.items() ]))

    df.to_csv("time_taken.csv")
    
if __name__ == "__main__":
    main(sys.argv[1:])
        
    