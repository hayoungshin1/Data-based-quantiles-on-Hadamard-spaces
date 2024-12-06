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
