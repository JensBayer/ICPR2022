from pathlib import Path

import sys
sys.path.append(str(Path('../WACV2022/PAN-PyTorch').resolve()))
from ops.transforms import GroupNormalize
from torchvision import transforms as T

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

norm = lambda x: (x-x.min()) / (x.max() - x.min()+1e-13)

groupNormalize = GroupNormalize(normalize.mean, normalize.std)