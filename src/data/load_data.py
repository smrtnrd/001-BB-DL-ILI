import pandas as pd

from delphi_epidata import Epidata

#import labels
def get_states():
    """return state label"""
    #create empty dic
    s = [] 
    try:
        # read the data
        with open("../src/labels/states.txt") as f :
            lines = f.readlines()
            #remove new lines at the end
            s = [ line.strip('\n') for line in lines ]
    except FileNotFoundError:
        return None
    else:
        return s

def get_ilinet_data(states, start, end):
    """
    return a dictionary of dataframe with the different epiweeks
    """
    ilinet_raw = {}
    for state in states:
        print("State {}".format(state))
        res = Epidata.ilinet(
            locations = state, #source
            epiweeks = [Epidata.range(start,end)]) #range 2009 to 2016
        if res['result'] == 1:
            print(res['result'], res['message'], len(res['epidata']))
            data = pd.DataFrame(res['epidata'])
            ilinet_raw[state] = data
        else:
            print("(-2, u'no success')")
    return ilinet_raw

def get_fluview_data(states, start, end):
    """
    return a dictionary of dataframe with the different epiweeks
    """
    ilinet_raw = {}
    for state in states:
        print("State {}".format(state))
        res = Epidata.fluview(
            regions = state, #source
            epiweeks = [Epidata.range(start,end)]) #range 2009 to 2016
        if res['result'] == 1:
            print(res['result'], res['message'], len(res['epidata']))
            data = pd.DataFrame(res['epidata'])
            ilinet_raw[state] = data
        else:
            print("(-2, u'no success')")
    return ilinet_raw
