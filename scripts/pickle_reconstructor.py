# coding: utf-8
import pickle
from ctlearn.ctapipe_plugin import CTLearnReconstructor
from ctapipe.instrument import SubarrayDescription
subarray = SubarrayDescription.from_hdf("/Users/tjarkmiener/DL_plugin_codesprint/ctlearn/test.dl1.h5")
ctlearn_reco = CTLearnReconstructor(subarray)
f = open("reconstructor.pkl", mode='wb')
pickle.dump(ctlearn_reco, f)
f.close()
