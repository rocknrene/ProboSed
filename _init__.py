# ProboSed Main Package Initializer
__version__ = "0.1.0"

# This allows your notebook to see the sub-modules
from . import slope
from . import transport
from . import core_ml
from . import utils

print(f"ProboSed v{__version__} loaded. Ready for VCD transcription.")
