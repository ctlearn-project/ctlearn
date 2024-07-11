from ctapipe.reco import Reconstructor
from ctapipe.containers import ArrayEventContainer, ReconstructedGeometryContainer, ReconstructedEnergyContainer
from ctapipe.reco.reconstructor import ReconstructionProperty, Reconstructor
from ctapipe.reco.stereo_combination import StereoMeanCombiner
from ctapipe.reco.utils import add_defaults_and_meta
from astropy.table import Table
import astropy.units as u
import numpy as np

class CTLearnReconstructor(Reconstructor):
    """CTLearn plugin Reconstructor"""
    prefix = "Hallo"
    property = ReconstructionProperty.ENERGY
    def __init__(self, subarray, **kwargs):
        super().__init__(subarray=subarray, **kwargs)
        self.stereo_combiner = StereoMeanCombiner(
            prefix=self.prefix,
            property=self.property,
            parent=self,
        )

    def __call__(self, event: ArrayEventContainer):
        """Foo"""
        event.dl2.geometry["PluginReconstructor"] = ReconstructedGeometryContainer()

    def predict_table(self, key, table: Table) -> dict[ReconstructionProperty, Table]:

        energy = np.random.lognormal(size=len(table)) * u.TeV
        is_valid = np.ones(len(table), dtype=bool)
        #energy[valid], is_valid[valid] = self._predict(key, table[valid])
        result = Table(
            {
                f"{self.prefix}_tel_energy": energy,
                f"{self.prefix}_tel_is_valid": is_valid,
            }
        )
        add_defaults_and_meta(
            result,
            ReconstructedEnergyContainer,
            prefix=self.prefix,
            add_tel_prefix=True,
        )
        return {ReconstructionProperty.ENERGY: result}
