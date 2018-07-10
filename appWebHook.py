# -*- coding:utf8 -*-

from __future__ import print_function
#from future.standard_library import install_aliases
#install_aliases()

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import json
import os

from flask import Flask
from flask import request
from flask import make_response

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.externals import joblib

# Flask app should start in global layout
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)

    print("Request:")
    # commented out by Naresh
    print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    # print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


def processRequest(req):
    print ("starting processRequest whit tensorFLow...",req.get("queryResult").get("action"))
    
    parameters = data.get("queryResult").get("outputContexts")[3].get("parameters")

    kmeans = joblib.load('model.pkl')
    result=kmeans.predictor([
        str(parameters.get("number_adults.original")),
        str(parameters.get("number_adults.original")),
        str(parameters.get("age.original")),
        ,
        ,
        
    ])
   
    #Here result search with destination trip!!!
    ## Code continue....
    
    data = {}
    res = makeWebhookResult(req)
    return res

def makeWebhookResult(data):
    print ("starting makeWebhookResult...")

    parameters = data.get("queryResult").get("outputContexts")[3].get("parameters")
    
    print(json.dumps(parameters,indent=4))

    speech = str(parameters.get("given-name")+
    " according to the given information, the destinations that better fit to your needs are: "
    )
    

    # + " for " 
    # + str(parameters.get("number_adults.original"))
    # + " adutls and "
    # + str(parameters.get("number_kids.original"))
    # + " childrens"

    print("Response:")
    print(speech)

    return {
        "fulfillmentText": speech,
        "source": "AiTravel-webhook"
    }


@app.route('/test', methods=['GET'])
def test():
    return  "Bienvenido Prueba Team RPA Maria !!"


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))

    print("Starting app on port %d" % port)

    app.run(debug=True, port=port, host='127.0.0.1')

