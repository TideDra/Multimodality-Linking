from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader

class DataLoaderX(DataLoader):
    '''
    A replacement to DataLoader which improves the pipeline performance.
    '''
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())