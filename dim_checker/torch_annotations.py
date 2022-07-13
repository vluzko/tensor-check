from dim_checker.types import *


def arange_annotate(args):
    pass


TorchType = Module({
    'arange': Dependent(arange_annotate)
})


