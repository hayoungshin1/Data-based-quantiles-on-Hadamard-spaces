full=tenfit.quadratic_form
full=torch.from_numpy(full)
x=full[area]
x=torch.flatten(x,0,2)
x=(x+torch.transpose(x,1,2))/2
x=x.float()
xinv=torch.linalg.inv(x)
xhalf=matrix_sqrt(x)
xhalfinv=torch.linalg.inv(xhalf)

torch.max(mag(xinv,log(xhalf,xhalfinv,torch.unsqueeze(torch.eye(x.shape[-1]),0))))
torch.max(mag(torch.unsqueeze(torch.eye(x.shape[-1]),0),log(torch.unsqueeze(torch.eye(x.shape[-1]),0),torch.unsqueeze(torch.eye(x.shape[-1]),0),x)))

from dipy.data import get_sphere
from dipy.viz import window, actor

betas=[0.98,0.8,0.6,0.4,0.2,0,0.2,0.4,0.6,0.8,0.98]
finalevals=np.zeros((len(betas),len(betas),1,3))
finalevecs=np.zeros((len(betas),len(betas),1,3,3))
for i in range(3):
    finalevecs[:,:,:,i,i]=1

finalcfa=np.zeros((len(betas),len(betas),1,3))
for j in range(4):
    if j==0:
        xi1='-'
        xi2='+'
    elif j==1:
        xi1=torch.tensor([[0,1,0],[1,0,0],[0,0,0]])/np.sqrt(2)
        xi2=-xi1
    elif j==2:
        xi1=torch.tensor([[0,0,1],[0,0,0],[1,0,0]])/np.sqrt(2)
        xi2=-xi1
    elif j==3:
        xi1=torch.tensor([[0,0,0],[0,0,1],[0,1,0]])/np.sqrt(2)
        xi2=-xi1
    xis=int((len(betas)-1)/2)*[xi1]+int((len(betas)+1)/2)*[xi2]
    quantiles=quantile(xinv, xhalf, xhalfinv, x, betas[0], xis[0])
    for i in range(len(betas)-1):
        quantiles=torch.concat((quantiles,quantile(xinv, xhalf, xhalfinv, x, betas[i+1], xis[i+1])),dim=0)
        print(j,i)
    L, V=torch.linalg.eig(quantiles)
    L=torch.real(L)
    L=torch.clamp(L,min=0)
    V=torch.real(V)
    qevals,indices=torch.sort(L,descending=True)
    qevecs=torch.zeros_like(V)
    for i in range(L.shape[0]):
        qevecs[i,:,:]=V[i,:,indices[i,:]]
    qevals=torch.unsqueeze(torch.unsqueeze(qevals,1),1)
    qevecs=torch.unsqueeze(torch.unsqueeze(qevecs,1),1)
    qevals=qevals.numpy()
    qevecs=qevecs.numpy()
    qFA = fractional_anisotropy(qevals)
    qFA[np.isnan(qFA)] = 0
    qFA = np.clip(qFA, 0, 1)
    qRGB = color_fa(qFA, qevecs)
    qcfa=qRGB
    qcfa /= cfa.max()
    if j==0:
        for k in range(len(betas)):
            finalevals[int((len(betas)-1)/2),k,0,:]=qevals[k,0,0,:]
            finalevecs[int((len(betas)-1)/2),k,0,:,:]=qevecs[k,0,0,:,:]
            finalcfa[int((len(betas)-1)/2),k,0,:]=qcfa[k,0,0,:]
    if j==1:
        for k in range(len(betas)):
            finalevals[k,k,0,:]=qevals[k,0,0,:]
            finalevecs[k,k,0,:,:]=qevecs[k,0,0,:,:]
            finalcfa[k,k,0,:]=qcfa[k,0,0,:]
    if j==2:
        for k in range(len(betas)):
            finalevals[k,int((len(betas)-1)/2),0,:]=qevals[k,0,0,:]
            finalevecs[k,int((len(betas)-1)/2),0,:,:]=qevecs[k,0,0,:,:]
            finalcfa[k,int((len(betas)-1)/2),0,:]=qcfa[k,0,0,:]
    if j==3:
        for k in range(len(betas)):
            finalevals[k,-k-1,0,:]=qevals[k,0,0,:]
            finalevecs[k,-k-1,0,:,:]=qevecs[k,0,0,:,:]
            finalcfa[k,-k-1,0,:]=qcfa[k,0,0,:]

sphere = get_sphere('repulsion724')
scene = window.Scene()
scene.add(actor.tensor_slicer(finalevals, finalevecs, scalar_colors=finalcfa, sphere=sphere, scale=0.3, norm=False))
scene.background((255,255,255)) # makes background white, remove to make black

window.show(scene)

scene.clear()
