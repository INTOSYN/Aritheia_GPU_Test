# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
__version__ = "0.5.0rc6"


def _reject_legacy_core(package_dir=None):
    """Do not silently import rc4 binary modules instead of the open rc5 sources."""
    from pathlib import Path
    root = Path(package_dir) if package_dir is not None else Path(__file__).parent
    modules = {"exact", "reference", "frozen", "native_smid", "probes"}
    collisions = sorted(
        path.name for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".so", ".pyd"}
        and path.name.split(".", 1)[0] in modules
    )
    if collisions:
        raise ImportError(
            "Legacy computeproof-core binary modules would shadow the open-source "
            "ComputeProof client: " + ", ".join(collisions) + ". "
            "Use a clean virtual environment, or uninstall computeproof-core and "
            "reinstall the open ComputeProof client in this environment. "
            "No packages were changed automatically."
        )


_reject_legacy_core()
