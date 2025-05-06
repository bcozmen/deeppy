from .nerf import Nerf
from .sane import Sane

algorithms =[a.__name__ for a in [Nerf,Sane]]