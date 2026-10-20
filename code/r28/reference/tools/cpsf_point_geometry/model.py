from tools.cpsf_pair_region.model import Model as Base

class Model(Base):
    def __init__(self,task,arm):
        super().__init__(task,'full' if arm=='shuffle_descriptor' else arm)
        self.descriptor_arm=arm
        self.use_descriptor=True

    def stems(self,g):
        if not self.use_descriptor or self.descriptor_arm=='skeleton':return super().stems(g)
        pf=g['pf_neighborhood']
        if self.descriptor_arm=='shuffle_descriptor':pf=pf[g['descriptor_permutation']]
        return super().stems(dict(g,pf=pf))
