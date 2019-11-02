import numpy as np

def check_backward(f,f_backward, inputs, epislon=1e-5):
    
    f0 = f(**inputs)
    for key in inputs:
        for i in inputs[key].size:
            inputs[key][i]
    