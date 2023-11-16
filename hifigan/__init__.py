from .hifigan import Vocoder


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


## The pretrained HIFIGAN vocoder could be accessed from: https://zenodo.org/records/7884686#.ZFBb6ezMKbh
