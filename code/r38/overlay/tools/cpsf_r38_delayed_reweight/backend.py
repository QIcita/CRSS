from .common import sha
from tools.cpsf_r38_dense.backend import Backend as DenseBackend, digest
from . import objective

class Backend(DenseBackend):
    def __init__(self, name, task, training):
        super().__init__(name, task, training)
        self.class_counts, self.class_weights = objective.configure(training)
        self.sources[__file__] = sha(__file__)
        self.sources[objective.__file__] = sha(objective.__file__)

    def set_epoch(self, epoch, epochs):
        assert epochs == 50
        super().set_epoch(epoch, epochs)
        objective.set_epoch(epoch)
