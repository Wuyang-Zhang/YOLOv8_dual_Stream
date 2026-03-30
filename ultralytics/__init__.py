# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.0"

from ultralytics.models import RTDETR, SAM, YOLO, YOLOWorld
from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

try:
    from ultralytics.data.explorer.explorer import Explorer
except ModuleNotFoundError:
    Explorer = None

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
