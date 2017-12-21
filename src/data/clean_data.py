# convert string into a date object
import sys  
import os


import time
import datetime as dt
from datetime import datetime

def dateconvert(Start_date):
    try:
        Start_date = int(Start_date)  
        Date_String = str(Start_date)
        dateconv = dt.datetime.strptime(Date_String,'%YYYY-%mm-%dd').strftime('%d/%m/%Y')
        return  dateconv
    except ValueError:
        return Start_date #or '' or whatever you want to return for NULL
  

def get_states():
    filepath = '/Users/bbuildman/Documents/Developer/GitHub/001-BB-DL-ILI/src/labels/states.txt'
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()
        
    states = []
    with open(filepath) as fp:
        for line in fp:
            states.append(line.rstrip('\n'))

    return states