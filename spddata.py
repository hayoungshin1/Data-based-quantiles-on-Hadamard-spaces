import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.data import get_fnames
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
data, affine = load_nifti(hardi_fname)
from dipy.data import get_sphere
from dipy.viz import window, actor

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)
from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2)
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(maskdata)

from dipy.reconst.dti import fractional_anisotropy, color_fa

FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0
FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)

sphere = get_sphere('repulsion724')


scene = window.Scene()

area=slice(13,43), slice(44,74), slice(28,29)

evals = tenfit.evals[area]
evecs = tenfit.evecs[area]

cfa = RGB[area]
biggest=cfa.max()

area=slice(13,43), slice(44,74), slice(28,29) # all of the tensors
#area=slice(33,40), slice(62,67), slice(28,29) # red subset

evals = tenfit.evals[area]
evecs = tenfit.evecs[area]

cfa = RGB[area]
cfa/=biggest

scene.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere, scale=0.5, norm=True))
scene.background((255,255,255)) # makes background white, remove to make black

window.show(scene)

window.record(scene, n_frames=1, out_path='wholetensors2.png', size=(2000, 2000))

scene.clear()
