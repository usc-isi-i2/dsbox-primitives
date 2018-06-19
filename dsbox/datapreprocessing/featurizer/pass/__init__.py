from pkgutil import extend_path
from .do_nothing import DoNothing
__path__ = extend_path(__path__, __name__)  # type: ignore

