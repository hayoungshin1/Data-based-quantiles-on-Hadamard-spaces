# approximation comparisons

m=64

betas=[0.2,0.4,0.6,0.8,0.98]
angles=[k*2*np.pi/m for k in range(m)]
xis=[torch.tensor([[np.cos(angles[k]),np.sin(angles[k])]]) for k in range(m)]
allquantiles=[]

np.random.seed(900)
x=np.random.normal(0,0.3,(100,2))
x=torch.Tensor(x)
for shape in [1,2]:
    if shape==2:
        x[:,1]/=4
    for t in ['p','d']:
        for approx in [1,2,3]:
            x=B2H(x)
            quantiles=quantile(x, 0, xis[0], t, approx)
            for i in range(len(betas)):
                for j in range(len(xis)):
                    quantiles=torch.concat((quantiles,quantile(x, betas[i], xis[j], t, approx)),dim=0)
                    print(i,j)
            allquantiles.append(quantiles)
            quantiles=H2B(quantiles)
            x=H2B(x)
            f = plt.figure(figsize=(7,7))
            ax = plt.gca()
            plt.scatter(x.cpu().numpy()[:,0], x.cpu().numpy()[:,1], s=30, c='0', marker = '.')
            plt.scatter(quantiles[0].cpu().numpy()[0], quantiles[0].cpu().numpy()[1], s=10, c='tab:blue', marker = '.')
            for i in range(len(betas)):
                plt.scatter(quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,0], quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,1], s=10, c='tab:blue', marker = '.')
                plt.plot(np.append(quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,0],quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,0][0])
            , np.append(quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,1],quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,1][0])
            , c='tab:blue')
            circle = plt.Circle((0, 0), 1, color='b', fill=False)
            ax.add_patch(circle)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            plt.axis('equal')
            plt.show(block=False)

for i in range(4):
    for j in range(2):
        for k in range(len(betas)):
            print('%.4f'%(torch.mean(newdist(allquantiles[3*i][(k*m+1):((k+1)*m+1)],allquantiles[3*i+j+1][(k*m+1):((k+1)*m+1)])).item()))

# permutation test

m=4

betas=[0.8]
angles=[k*2*np.pi/m for k in range(m)]
xis=[torch.tensor([[np.cos(angles[k]),np.sin(angles[k])]]) for k in range(m)]

np.random.seed(900)
x1=np.random.normal(0,0.3,(100,2))
x1=torch.Tensor(x1)
x2=x1.detach().clone()
x2[:,1]/=4
x1=B2H(x1)
x2=B2H(x2)
x1quantiles=quantile(x1, 0, xis[0], 'd', 1)
x2quantiles=quantile(x2, 0, xis[0], 'd', 1)
medstat=torch.sum(newdist(x1quantiles,x2quantiles)).item()
for i in range(len(betas)):
    for j in range(len(xis)):
        x1quantiles=torch.concat((x1quantiles,quantile(x1, betas[i], xis[j], 'd', 1)),dim=0)
        x2quantiles=torch.concat((x2quantiles,quantile(x2, betas[i], xis[j], 'd', 1)),dim=0)

quantstat=torch.sum(newdist(x1quantiles,x2quantiles)).item()
print(medstat,quantstat)

medsum=0
quantsum=0
totalx=torch.concat((x1,x2),0)
reps=500
for k in range(reps):
    permindices=np.random.choice(totalx.shape[0],x1.shape[0],replace=False)
    permx1=totalx[permindices,:]
    permx2=np.delete(totalx,permindices,axis=0)
    permx1quantiles=quantile(permx1, 0, xis[0], 'd', 1)
    permx2quantiles=quantile(permx2, 0, xis[0], 'd', 1)
    permmedstat=torch.sum(newdist(permx1quantiles,permx2quantiles)).item()
    for i in range(len(betas)):
        for j in range(len(xis)):
            permx1quantiles=torch.concat((permx1quantiles,quantile(permx1, betas[i], xis[j], 'd', 1)),dim=0)
            permx2quantiles=torch.concat((permx2quantiles,quantile(permx2, betas[i], xis[j], 'd', 1)),dim=0)
    permquantstat=torch.sum(newdist(permx1quantiles,permx2quantiles)).item()
    medsum+=(permmedstat>=medstat)
    quantsum+=(permquantstat>=quantstat)
    print(k,medsum,quantsum)

print(medsum/reps,quantsum/reps)
