#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:43:17 2017

@author: captain
"""
import fredapi
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from dateutil.relativedelta import relativedelta
utc=pytz.UTC
with open('api_keys.json','r') as f:
    api_keys=json.load(f)
    fk=api_keys.get('fred',None)

def get_fred_data(fred_ticker,name_out):
    fred=fredapi.Fred(api_key=fk)
    ts=fred.get_series(fred_ticker)
    return pd.DataFrame(ts,columns=[name_out])


def get_fred_eco_data(fred_eco_ticker,name_out,avail_gap):
    """
    most of the eco releases are available with substantial gap.
    one needs to figure out what the gap is and pass in an upper bound
    """
    data=get_fred_data(fred_eco_ticker,name_out)
    add_days=relativedelta(days=avail_gap)
    data.loc[:,'avail_date']=pd.to_datetime(data.index)
    data.loc[:,'avail_date']=data.avail_date.apply(lambda x: x+add_days)
    data.set_index('avail_date',inplace=True)
    return data    

