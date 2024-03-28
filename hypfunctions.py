import numpy as np
import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

def ip(v1, v2):
    """
    v1: batch of vectors (b,n+1) or (1,n+1)
    v2: batch of vectors (b,n+1) or (1,n+1)
    out: (b)
    """
    copy=v1.detach().clone()
    copy[:,0]=-copy[:,0]
    out=torch.sum(copy*v2, dim=1)
    return out

def mag(v1):
    """
    v1: batch of vectors (b,n+1)
    out: (b)
    """
    sq=ip(v1, v1)
    out=torch.sqrt(torch.clamp(sq,min=0)) # ensures out is real
    return out

def exp(p, v):
    """
    p: point in H^n (1,n+1)
    v: vector in T_pM (1,n+1)
    out: (1,n+1)
    """
    p=p/torch.sqrt(-ip(p,p)) # reprojects p onto the manifold, for precision
    theta=mag(v)
    if theta==0:
        out=p
    else:
        unitv=v/theta
        out=torch.cosh(theta)*p+torch.sinh(theta)*unitv
    out=out/torch.sqrt(-ip(out,out)) # reprojects out onto the manifold, for precision
    return out

def log(p, x):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    out: each log_p(x) (b,n+1)
    """
    p=p/torch.sqrt(-ip(p,p)) # reprojects p onto the manifold, for precision
    x=x/torch.unsqueeze(torch.sqrt(-ip(x,x)),dim=1) # reprojects x onto the manifold, for precision
    a=ip(p,x)
    a=torch.clamp(a,max=-1) # ensures -a is at least 1
    theta=torch.acosh(-a)
    v=x+torch.matmul(torch.unsqueeze(ip(p,x),1),p)
    t=torch.unsqueeze(mag(v),1)
    unitv=v/t
    out=torch.unsqueeze(theta,1)*unitv
    return out

def alog(x, p):
    """
    x: batch of points in H^n (b,n+1)
    p: point in H^n (1,n+1)
    out: each log_x(p) (b,n+1)
    """
    p=p/torch.sqrt(-ip(p,p)) # reprojects p onto the manifold, for precision
    x=x/torch.unsqueeze(torch.sqrt(-ip(x,x)),dim=1) # reprojects x onto the manifold, for precision
    a=ip(p,x)
    a=torch.clamp(a,max=-1) # ensures -a is at least 1
    theta=torch.acosh(-a)
    v=p+torch.unsqueeze(ip(x,p),1)*x
    t=torch.unsqueeze(mag(v),1)
    unitv=v/t
    out=torch.unsqueeze(theta,1)*unitv
    return out

def newlog(y, z):
    """
    y: batch of points in H^n (b,n+1)
    z: point in H^n (b,n+1)
    out: each log_y(z) (b,n+1)
    """
    y=y/torch.unsqueeze(torch.sqrt(-ip(y,y)),dim=1) # reprojects y onto the manifold, for precision
    z=z/torch.unsqueeze(torch.sqrt(-ip(z,z)),dim=1) # reprojects z onto the manifold, for precision
    a=ip(y,z)
    a=torch.clamp(a,max=-1) # ensures -a is at least 1
    theta=torch.acosh(-a)
    v=z+torch.unsqueeze(ip(y,z),1)*y
    t=torch.unsqueeze(mag(v),1)
    unitv=v/t
    out=torch.unsqueeze(theta,1)*unitv
    return out

def dist(p, x):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    out: (b)
    """
    out=mag(log(p, x))
    return out

def adist(x, p):
    """
    x: batch of points in H^n (b,n+1)
    p: point in H^n (1,n+1)
    out: (b)
    """
    out=mag(alog(x, p))
    return out

def newdist(y, z):
    """
    y: batch of points in H^n (b,n+1)
    z: point in H^n (b,n+1)
    out: (b)
    """
    out=mag(newlog(y, z))
    return out

def direct(p,xi):
    """
    p: point in H^n (1,n+1)
    xi: point in S^(n-1), the boundary at infinite (1,n)
    out: unit vector in T_pM in direction of xi (1,n+1)
    """
    p=p/torch.sqrt(-ip(p,p)) # reprojects p onto the manifold, for precision
    xi=torch.concat((torch.tensor([[1]]),xi),dim=1)
    out=xi+ip(p,xi)*p
    out=out/mag(out)
    return out

def adirect(x,xi):
    """
    x: batch of points in H^n (b,n+1)
    xi: point in S^(n-1), the boundary at infinite (1,n)
    out: unit vectors in T_xM in direction of xi (b,n+1)
    """
    xi=torch.concat((torch.tensor([[1]]),xi),dim=1)
    out=xi+torch.unsqueeze(ip(x,xi),1)*x
    out=out/torch.unsqueeze(mag(out),1)
    return out

def loss(p, x, beta, xi, t):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    beta: number in [0,1)
    xi: point in S^(n-1), the boundary at infinite (1,n)
    t: 'p' or 'd'
    """
    if t=='p':
        out=torch.mean(dist(p,x)+ip(beta*direct(p,xi),log(p,x)))
    elif t=='d':
        out=torch.mean(adist(x,p)-ip(beta*adirect(x,xi),alog(x,p)))
    return out

def pt(x, v, p):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    v: batch of vectors in tangent spaces at x (b,n+1)
    out: v parallel transported to T_pH^n (b,n+1)
    """
    out=v-torch.unsqueeze(ip(alog(x,p),v)/(adist(x,p))**2,1)*(log(p,x)+alog(x,p))
    return out

def grad(p, x, beta, xi, t, approx):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    beta: number in [0,1)
    xi: point in S^(n-1), the boundary at infinite (1,n)
    t: 'p' for parameter-based quantiles or 'd' for data-based quantiles
    approx: 1 for true gradient, 2 uses parallel transport of xi_x, and 3 uses xi_p
    out: gradient (1,n+1)
    """
    dpx=dist(p,x)
    lpx=log(p,x)
    unitpx=lpx/torch.unsqueeze(dpx,1)
    if approx==1:
        if t=='p':
            xip=direct(p,xi)
            cothdpx=torch.cosh(dpx)/torch.sinh(dpx)
            out=-unitpx-beta*(torch.unsqueeze((1-dpx*cothdpx)*ip(xip,unitpx)+dpx,1)*unitpx+torch.unsqueeze(dpx*(cothdpx-ip(xip,unitpx)),1)*xip)
            for j in [i for i, x in enumerate(dpx<1e-20) if x]: # just set to 0 when p and x_j are very close
                out[j,:]=-beta*xip[0,:]
        elif t=='d':
            lxp=alog(x,p)
            xix=adirect(x,xi)
            unitxp=lxp/torch.unsqueeze(dpx,1)
            new=pt(x,xix,p)
            out=-unitpx-beta*(torch.unsqueeze(dpx/torch.sinh(dpx),1)*new+torch.unsqueeze((dpx/torch.sinh(dpx)-1)*ip(xix,unitxp),1)*unitpx)
            for j in [i for i, x in enumerate(dpx<1e-20) if x]: # just set to 0 when p and x_j are very close
                out[j,:]=-beta*new[j,:]
    elif approx==2:
        lxp=alog(x,p)
        xix=adirect(x,xi)
        new=pt(x,xix,p)
        out=-unitpx-beta*new
        for j in [i for i, x in enumerate(dpx<1e-20) if x]: # just set to 0 when p and x_j are very close
            out[j,:]=-beta*new[j,:]
    elif approx==3:
        xip=direct(p,xi)
        out=-unitpx-beta*xip
        for j in [i for i, x in enumerate(dpx<1e-20) if x]: # just set to 0 when p and x_j are very close
            out[j,:]=-beta*xip[0,:]
    out=torch.mean(out,dim=0,keepdim=True)
    return out

def quantile(x, beta, xi, t, approx, tol=1e-100):
    """
    x: batch of points in H^n (b,n+1)
    beta: number in [0,1)
    xi: point in S^(n-1), the boundary at infinite (1,n)
    t: 'p' for parameter-based quantiles or 'd' for data-based quantiles
    approx: 1 for true gradient, 2 uses parallel transport of xi_x, and 3 uses xi_p
    out: (beta, xi)th-quantile (1,n+1)
    """
    x=x/torch.unsqueeze(torch.sqrt(-ip(x,x)),1) # reprojects x onto the manifold, for precision
    xi=xi/torch.sqrt(torch.sum(xi*xi)) # ensures xi is on S^n
    current_p=torch.unsqueeze(torch.concat((torch.ones(1),torch.zeros(x.shape[1]-1))),0) # initial estimate for quantile
    old_p=current_p.detach().clone()
    current_loss=loss(current_p,x,beta,xi,t)
    lr=0.001
    step=-grad(current_p,x,beta,xi,t,approx)
    step/=mag(step)
    count=0
    while lr>1e-100 and count<1000:
        new_p=exp(current_p,lr*step).float()
        new_loss=loss(new_p,x,beta,xi,t)
        if (new_loss<=current_loss):
            old_p=current_p
            current_p=new_p
            current_loss=new_loss
            step=-grad(current_p,x,beta,xi,t,approx)
            step/=mag(step)
            lr=1.1*lr # try to speed up convergence by increasing learning rate
        else:
            lr=lr/2
            count+=1
    out=current_p
    return out

def H2B(p):
    """
    p: batch of points in H^n (b,n+1)
    out: p sent to the Poincare ball B^n (b,n)
    """
    out=p[:,1:]/torch.unsqueeze(p[:,0]+1,1)
    return out

def B2H(x):
    '''
    A map from B^n to H^n
    x:      torch.tensor whose size = (b, n)
    out:    torch.tensor whose size = (b, n + 1)
    '''
    norm_square = (torch.norm(x, dim = 1) ** 2).unsqueeze(dim=1)
    out = torch.cat([(1 + norm_square)/(1 - norm_square), 2 * x / (1 - norm_square)], dim = 1)
    return out
