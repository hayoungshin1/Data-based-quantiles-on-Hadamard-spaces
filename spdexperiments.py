full=tenfit.quadratic_form
full=torch.from_numpy(full)
x=full[area]
x=torch.flatten(x,0,2)
x=(x+torch.transpose(x,1,2))/2
x=x.float()
xinv=torch.linalg.inv(x)
xhalf=matrix_sqrt(x)
xhalfinv=torch.linalg.inv(xhalf)
identity=2*torch.tensor([[[1,0,0],[0,1,0],[0,0,1]]])/2

# visualizations

betas=[0.98,0.8,0.6,0.4,0.2,0,0.2,0.4,0.6,0.8,0.98]
finalevals=np.zeros((len(betas),len(betas),1,3))
finalevecs=np.zeros((len(betas),len(betas),1,3,3))
for i in range(3):
    finalevecs[:,:,:,i,i]=1


finalcfa=np.zeros((len(betas),len(betas),1,3))

for j in range(4):
    if j==0:
        xi1=torch.tensor([[[1,0,0],[0,1,0],[0,0,1]]])/np.sqrt(3)
        xi2=-xi1
    elif j==1:
        xi1=torch.tensor([[[0,1,0],[1,0,0],[0,0,0]]])/np.sqrt(2)
        xi2=-xi1
    elif j==2:
        xi1=torch.tensor([[[0,0,1],[0,0,0],[1,0,0]]])/np.sqrt(2)
        xi2=-xi1
    elif j==3:
        xi1=torch.tensor([[[0,0,0],[0,0,1],[0,1,0]]])/np.sqrt(2)
        xi2=-xi1
    xis=int((len(betas)-1)/2)*[xi1]+int((len(betas)+1)/2)*[xi2]
    quantiles=quantile(xinv, xhalf, xhalfinv, x, betas[0], identity, identity, xis[0])
    for i in range(len(betas)-1):
        quantiles=torch.concat((quantiles,quantile(xinv, xhalf, xhalfinv, x, betas[i+1], identity, identity, xis[i+1])),dim=0)
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
scene.background((255,255,255))

window.show(scene)

window.record(scene, n_frames=1, out_path='wholequantiles4.png', size=(2000, 2000))

scene.clear()

# distributional characteristic measures

dispersion=[]
skewness=[]
kurtosis=[]
supdispersion=[]
avedispersion=[]
supskewness=[]
aveskewness=[]
supkurtosis=[]
avekurtosis=[]
sasymmetry=[]

m=96

basis=torch.ones(6,3,3)
basis[0]=torch.tensor([[1,0,0],[0,0,0],[0,0,0]])
basis[1]=torch.tensor([[0,0,0],[0,1,0],[0,0,0]])
basis[2]=torch.tensor([[0,0,0],[0,0,0],[0,0,1]])
basis[3]=torch.tensor([[0,1,0],[1,0,0],[0,0,0]])/np.sqrt(2)
basis[4]=torch.tensor([[0,0,1],[0,0,0],[1,0,0]])/np.sqrt(2)
basis[5]=torch.tensor([[0,0,0],[0,0,1],[0,1,0]])/np.sqrt(2)
np.random.seed(1)
vecs=torch.normal(mean=torch.zeros(m,6),std=torch.ones(m,6))
vecs=vecs/torch.sqrt(torch.sum(vecs*vecs,dim=1,keepdim=True))
radial=torch.sum(torch.unsqueeze(torch.unsqueeze(vecs,2),2)*torch.unsqueeze(basis,0),1)
radial=torch.concat((radial,-radial),dim=0)

for moment in ['other', 'kurtosis']:
    for extreme in ['no', 'yes']:
        if extreme=='no':
            if moment=='kurtosis':
                betas=torch.tensor([0.2,0.8])
            else:
                betas=torch.tensor([0.5])
        if extreme=='yes':
            if moment=='kurtosis':
                betas=torch.tensor([0.2,0.98])
            else:
                betas=torch.tensor([0.98])
        quantiles=quantile(xinv, xhalf, xhalfinv, x, 0, identity, identity, torch.unsqueeze(radial[0,:,:],0)) # frechet median
        median=quantiles.detach().clone()
        medianinv=torch.linalg.inv(median)
        medianhalf=matrix_sqrt(median)
        medianhalfinv=torch.linalg.inv(medianhalf)
        xis=pt(log(identity,identity,median),identity,identity,radial) # xi at frechet median
        for i in range(len(betas)):
            for j in range(len(xis)):
                quantiles=torch.concat((quantiles,quantile(xinv, xhalf, xhalfinv, x, betas[i].item(), medianhalf, medianhalfinv, torch.unsqueeze(xis[j,:,:],0))),dim=0)
                print(i,j)
        lift=log(medianhalf,medianhalfinv,quantiles)
        interranges=torch.zeros(int(len(xis)/2))
        for j in range(int(len(xis)/2)):
            interranges[j]=mag(medianinv,torch.unsqueeze(lift[j+1,:,:]-lift[j+1+int(len(xis)/2),:,:],0))
        supinterrange=torch.max(interranges).item()
        aveinterrange=torch.mean(interranges).item()
        opps=torch.zeros(int(len(xis)/2),3,3)
        if moment=='other':
            supdispersion.append(supinterrange)
            avedispersion.append(aveinterrange)
            for j in range(int(len(xis)/2)):
                opps[j,:,:]=lift[j+1,:,:]+lift[j+1+int(len(xis)/2),:,:]
            supskewness.append(torch.max(mag(medianinv,opps)).item()/supinterrange)
            aveskewness.append((mag(medianinv,torch.unsqueeze(torch.mean(opps,0)/2,0))/aveinterrange).item())
            sasymmetry.append(torch.abs(torch.log(torch.max(mag(medianinv,lift[1:,:,:]))/torch.min(mag(medianinv,lift[1:,:,:])))).item())
        elif moment=='kurtosis':
            for j in range(int(len(xis)/2)):
                opps[j,:,:]=lift[j+1+len(xis),:,:]-lift[j+1+3*int(len(xis)/2),:,:]
            supkurtosis.append(torch.max(mag(medianinv,opps)).item()/supinterrange)
            avekurtosis.append(torch.mean(mag(medianinv,opps)).item()/aveinterrange)
        print('supdispersion:')
        print(supdispersion)
        print('avedispersion:')
        print(avedispersion)
        print('supskewness:')
        print(supskewness)
        print('aveskewness:')
        print(aveskewness)
        print('supkurtosis:')
        print(supkurtosis)
        print('avekurtosis:')
        print(avekurtosis)
        print('sasymmetry:')
        print(sasymmetry)
