import numpy as np

def qrot(q, v):    
    uv = np.cross(q[:3], v)
    uuv = np.cross(q[:3], uv)
    vr = v + 2 * (q[3] * uv + uuv)
    return vr

def qrot_backward(q, v, vr_b):
    uv = np.cross(q[:3], v)
    v_b = vr_b.copy()
    q_b = np.zeros((4))
    q_b[3] = np.sum(vr_b * uv)
    uv_b = vr_b * q[3]
    uuv_b = vr_b.copy()
    q_b[:3] = np.sum(np.cross(uuv_b,uv), axis=0)
    return q_b,v_b

def normalize(x,axis=-1):
    n2 = np.sum(x**2,axis=axis)
    n = np.sqrt(n2)
    xn=x/np.expand_dims(n,axis)
    return xn

def normalize_backward(x, xn_b, axis=-1 ):
    n2 = np.sum(x**2,axis = axis)
    n = np.sqrt(n2)
    inv_n = 1/n
    
    n_b = -np.sum(xn_b * x,axis=1) * (inv_n**2)
    n2_b = 0.5*n_b/n
    x_b = (xn_b- x*np.expand_dims(n_b,axis))*np.expand_dims(inv_n,axis)
    return x_b

def cross_backward(u,v,c_b):
    u_b= np.cross(c_b, u)
    v_b= np.cross(u,c_b)
    return u_b,v_b
