"""verifiers v1 taskset plugin entry point: ``taskset.id = "primebeaker-articulated-harness"``.

v1 plugin ids must name a top-level module, so this re-exports the taskset from
``primebeaker.environments.articulated_harness_v1``.
"""

from primebeaker.environments.articulated_harness_v1 import *  # noqa: F403
from primebeaker.environments.articulated_harness_v1 import __all__  # noqa: F401
