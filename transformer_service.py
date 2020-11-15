import bentoml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bentoml.adapters import JsonInput
from bentoml.service.artifacts.common import PickleArtifact
import torch
import math
import warnings
import numpy as np

@bentoml.env(pip_packages=["transformers==3.1.0", "torch==1.6.0"])
@bentoml.artifacts([PickleArtifact('bartModel')])
class TransformerService(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, payload):
        # Initialize labels
        risk_labels = ["Data Collection", 
                            "Data Privacy", 
                            "Data Processes", 
                            "Data Modelling", 
                            "Data Encryption",
                            "Data Security", 
                            "Not Related to Data"]
                
        # Natural Language Inference Premise inputs
        nli_template_risk_label = "The risk factor of this text is on {}."  
        
        # Get text paypload
        response = payload["response"]
        
        # JSON output
        risk_dict = self.artifacts.bartModel(response, risk_labels, hypothesis_template=nli_template_risk_label)
        
        return risk_dict
