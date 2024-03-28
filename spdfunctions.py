import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

def ip(pinv, v1, v2):
    """
    pinv: one or a batch of p^{-1}, directly inputed to save time (1,n,n), or (b,n,n)
    v1, v2: batches of tangent vectors in T_pM (b,n,n)
    out: inner products (b)
    """
    t1=torch.matmul(pinv,v1)
    t2=torch.matmul(pinv,v2)
    out=torch.matmul(t1,t2).diagonal(offset=0,dim1=-1,dim2=-2).sum(-1)
    return out

def mag(pinv, v1):
    """
    pinv: one or a batch of p^{-1}, directly inputed to save time (1,n,n), or (b,n,n)
    v1: batch of tangent vectors in T_pM (b,n,n)
    out: size of each v1 (b)
    """
    sq=ip(pinv, v1, v1)
    out=torch.sqrt(torch.clamp(sq,min=0)) # ensures out is real
    return out

def matrix_exp(A):
    """
    A: batch of symmetric positive definite matrices (b,n,n)
    out: matrix exponentials of those matrices (b,n,n)
    """
    A=(A+torch.transpose(A,-2,-1))/2
    L, V=torch.linalg.eig(A)
    L=torch.real(L) # ensure realness
    V=torch.real(V) # ensure realness
    out=torch.matmul(torch.matmul(V,torch.diag_embed(torch.exp(L))),torch.linalg.inv(V))
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def matrix_log(A):
    """
    A: batch of symmetric positive definite matrices (b,n,n)
    out: matrix logs of those matrices (b,n,n)
    """
    A=(A+torch.transpose(A,-2,-1))/2
    L, V=torch.linalg.eig(A)
    L=torch.real(L) # ensure realness
    L=torch.clamp(L,min=0) # ensure non-negativeness
    V=torch.real(V) # ensure realness
    out=torch.matmul(torch.matmul(V,torch.diag_embed(torch.log(L))),torch.linalg.inv(V))
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def matrix_sqrt(A):
    """
    A: batch of symmetric positive definite matrices (b,n,n)
    out: matrix square roots of those matrices (b,n,n)
    """
    A=(A+torch.transpose(A,-2,-1))/2
    L, V=torch.linalg.eig(A)
    L=torch.real(L) # ensure realness
    L=torch.clamp(L,min=0) # ensure non-negativeness
    V=torch.real(V) # ensure realness
    out=torch.matmul(torch.matmul(V,torch.diag_embed(torch.sqrt(L))),torch.linalg.inv(V))
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def exp(phalf, phalfinv, v):
    """
    phalf: p^{1/2}, directly inputed to save time (1,n,n)
    phalfinv: p^{-1/2}, directly inputed to save time (1,n,n)
    v: batch of tangent vectors in each T_pM (b,n,n)
    out: each exp_p(v) (b)
    """
    v=(v+torch.transpose(v,-2,-1))/2
    out=matrix_exp(torch.matmul(torch.matmul(phalfinv,v),phalfinv))
    out=torch.matmul(torch.matmul(phalf,out),phalf)
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def log(xhalf, xhalfinv, p):
    """
    xhalf: batch of x^{1/2}, directly inputed to save time (b,n,n)
    xhalfinv: batch of x^{-1/2}, directly inputed to save time (b,n,n)
    p: point in P_n (1,n,n)
    out: each log_x(p) (b)
    """
    p=(p+torch.transpose(p,-2,-1))/2
    out=matrix_log(torch.matmul(torch.matmul(xhalfinv,p),xhalfinv))
    out=torch.matmul(torch.matmul(xhalf,out),xhalf)
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def direct(x, xi):
    """
    x: batch of points in P_n (b,n,n)
    xi: either a unit vector in T_IM (n,n), where I is the nxn identity, or '+' or '-', which represent the
    unit vectors I/sqrt(n), -I/sqrt(n) in T_IM
    out: each xi_x (b,n,n)
    """
    if xi=='+':
        out=x/np.sqrt(x.shape[-1])
    elif xi=='-':
        out=-x/np.sqrt(x.shape[-1])
    else:
        big=10
        inner=exp(torch.unsqueeze(torch.eye(x.shape[-1]),0),torch.unsqueeze(torch.eye(x.shape[-1]),0),big*xi)
        out=log(xhalf,xhalfinv,inner)
        out/=torch.unsqueeze(torch.unsqueeze(mag(xinv,out),-1),-1)
    return out

def loss(lxp, xinv, beta, xix):
    """
    lxp: each log_x(p), directly inputed to save time (b,n,n)
    xinv: batch of x^{-1}, directly inputed to save time (b,n,n)
    beta: real number in [0,1)
    xix: each xi_x, directly inputed to save time (b,n,n)
    out: data-based loss (b)
    """
    out=torch.mean(mag(xinv,lxp)-ip(xinv,beta*xix,lxp))
    return out

def pt(lxp, xhalf, xhalfinv, v):
    """
    lxp: each log_x(p), directly inputed to save time (b,n,n)
    xhalf: batch of x^{1/2}, directly inputed to save time (b,n,n)
    xhalfinv: batch of x^{-1/2}, directly inputed to save time (b,n,n)
    v: batch of tangent vectors in each T_xM (b,n,n)
    out: parallel transport from x to p (b,n,n)
    """
    big=torch.matmul(torch.matmul(xhalfinv,lxp),xhalfinv)/2
    big=torch.matmul(torch.matmul(xhalf,matrix_exp(big)),xhalfinv)
    out=torch.matmul(torch.matmul(big,v),torch.transpose(big,-2,-1))
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def grad(lxp, xhalf, xhalfinv, p, x, beta, xix):
    """
    lxp: each log_x(p), directly inputed to save time (b,n,n)
    xhalf: batch of x^{1/2}, directly inputed to save time (b,n,n)
    xhalfinv: batch of x^{-1/2}, directly inputed to save time (b,n,n)
    p: point in P_n (1,n,n)
    x: batch of points in P_n (b,n,n)
    beta: real number in [0,1)
    xix: each xi_x, directly inputed to save time (b,n,n)
    out: approximate gradient in T_pM (1,n,n)
    """
    pinv=torch.linalg.inv(p)
    phalf=matrix_sqrt(p)
    phalfinv=torch.linalg.inv(phalf)
    lpx=log(phalf,phalfinv,x)
    dpx=mag(pinv,lpx)
    unitpx=lpx/torch.unsqueeze(torch.unsqueeze(dpx,-1),-1)
    new=pt(lxp,xhalf,xhalfinv,xix)
    out=-unitpx-beta*new
    for j in [i for i, x in enumerate(dpx<1e-20) if x]: # just set to 0 when p and x_j are very close
        out[j,:,:]=new[j,:,:]
    out=torch.mean(out,dim=0,keepdim=True)
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def quantile(xinv, xhalf, xhalfinv, x, beta, xi, tol=1e-100):
    """
    xinv: batch of x^{-1}, directly inputed to save time (b,n,n)
    xhalf: batch of x^{1/2}, directly inputed to save time (b,n,n)
    xhalfinv: batch of x^{-1/2}, directly inputed to save time (b,n,n)
    x: batch of points in P_n (b,n,n)
    beta: real number in [0,1)
    xi: either a unit vector in T_IM (n,n), where I is the nxn identity, or '+' or '-', which represent the
    unit vectors I/sqrt(n), -I/sqrt(n) in T_IM
    out: (beta,xi)-quantile (1,n,n)
    """
    xix=direct(x,xi)
    current_p=torch.unsqueeze(torch.eye(x.shape[-1]),0) # initial estimate for quantile
    current_pinv=torch.linalg.inv(current_p)
    current_phalf=matrix_sqrt(current_p)
    current_phalfinv=torch.linalg.inv(current_phalf)
    current_lxp=log(xhalf,xhalfinv,current_p)
    old_p=current_p.detach().clone()
    current_loss=loss(current_lxp,xinv,beta,xix)
    lr=0.001
    step=-grad(current_lxp,xhalf,xhalfinv,current_p,x,beta,xix)
    step/=mag(current_pinv,step)
    count=0
    while lr>tol and count<1000:
        new_p=exp(current_phalf,current_phalfinv,lr*step).float()
        new_pinv=torch.linalg.inv(new_p)
        new_phalf=matrix_sqrt(new_p)
        new_phalfinv=torch.linalg.inv(new_phalf)
        new_lxp=log(xhalf,xhalfinv,new_p)
        new_loss=loss(new_lxp,xinv,beta,xix)
        if (new_loss<=current_loss):
            old_p=current_p
            current_p=new_p
            current_pinv=new_pinv
            current_phalf=new_phalf
            current_phalfinv=new_phalfinv
            current_lxp=new_lxp
            current_loss=new_loss
            step=-grad(current_lxp,xhalf,xhalfinv,current_p,x,beta,xix)
            step/=mag(current_pinv,step)
            lr=1.1*lr # try to speed up convergence by increasing learning rate
        else:
            lr=lr/2
            count+=1
    out=current_p
    return out
