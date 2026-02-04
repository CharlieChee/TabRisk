"""Model interfaces and implementations."""

from synthgen.models.base import BaseModel as BaseGeneratorModel
from synthgen.models.sdv_ctgan import SDVCTGANModel

__all__ = ["BaseGeneratorModel", "SDVCTGANModel"]
