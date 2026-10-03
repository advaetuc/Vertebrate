# Third-party attribution — P0

This project uses Ultralytics YOLO 8.3.203 and the YOLO11n pose weights under
**AGPL-3.0**. The transitive `ultralytics-thop` package also declares AGPL-3.0.
These components are not MIT-licensed. Preserve their notices and comply with
the applicable license terms when modifying or distributing a combined work.
Running locally does not change their license. This attribution document does
not assign a license to the user's original project code.

Primary references: [Ultralytics pinned license](https://github.com/ultralytics/ultralytics/blob/v8.3.203/LICENSE),
[Ultralytics model/software licensing](https://www.ultralytics.com/license),
[official pose weight release](https://github.com/ultralytics/assets/releases/tag/v8.3.0),
[GNU AGPL-3.0 text](https://www.gnu.org/licenses/agpl-3.0.html).

## Direct dependencies and model

| Component | Installed version | Attribution / license |
|---|---|---|
| Ultralytics / YOLO11n pose weights | 8.3.203 / v8.3.0 asset release | Ultralytics; AGPL-3.0 |
| PyTorch | 2.8.0+cpu | PyTorch contributors; BSD-3-Clause, plus bundled notices |
| torchvision | 0.23.0+cpu | PyTorch/torchvision contributors; BSD (see installed LICENSE) |
| NumPy | 2.2.6 | NumPy Developers; BSD-3-Clause, plus bundled numerical-library notices |
| opencv-python | 4.12.0.88 | OpenCV contributors; metadata declares Apache 2.0; preserve LICENSE.txt and LICENSE-3RD-PARTY.txt |
| PySide6 / Qt | 6.9.2 | The Qt Company and contributors; metadata: LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only; commercial terms are a separate option |
| lap | 0.5.12 | lap contributors; BSD-2-Clause |
| PyYAML | 6.0.2 | PyYAML contributors; MIT |
| pytest (development) | 8.4.2 | pytest contributors; MIT |
| Python interpreter | 3.12.0 | Python Software Foundation; PSF license and bundled notices |

Upstream references: [PyTorch](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE),
[torchvision](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE),
[NumPy](https://github.com/numpy/numpy/blob/v2.2.6/LICENSE.txt),
[OpenCV Python packaging](https://github.com/opencv/opencv-python),
[Qt for Python licenses](https://doc.qt.io/qtforpython-6/licenses.html),
[lap](https://github.com/rathaROG/lap), [PyYAML](https://github.com/yaml/pyyaml),
[pytest](https://github.com/pytest-dev/pytest).

Qt modules and bundled third-party libraries may have additional or different
terms. P0 imports QtCore, QtGui and QtWidgets; it does not use PyQt. This table is
an inventory, not a claim that every bundled Qt module has identical licensing.

## Remaining resolved distributions

Version-specific license metadata, declared license file names and upstream
project URLs for **all 48 installed distributions** (including the application)
are preserved verbatim in [environment.json](../reports/environment.json).
The [CPU lock](../requirements-win-cpu.lock) pins every third-party version.
The following summaries follow the installed metadata; a generic BSD declaration
is left generic rather than inventing a clause count.

| Distribution(s) | Declared license / notice |
|---|---|
| certifi | MPL-2.0 |
| charset-normalizer, filelock, fonttools, iniconfig | MIT; fonttools also includes LICENSE.external |
| colorama, cycler, Jinja2, kiwisolver, mpmath, sympy | BSD; consult installed full license text |
| contourpy, fsspec, idna, MarkupSafe, networkx, psutil | BSD-3-Clause |
| matplotlib | Matplotlib license agreement (PSF-style), with bundled notices |
| packaging | Apache-2.0 OR BSD-2-Clause |
| pillow | MIT-CMU, with bundled imaging-library notices |
| pip, pluggy, polars, polars-runtime-32, pyparsing | MIT |
| Pygments | BSD-2-Clause |
| PySide6_Addons, PySide6_Essentials, shiboken6 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only per metadata; module-specific Qt notices apply |
| python-dateutil | Dual BSD / Apache license; consult installed LICENSE |
| requests | Apache-2.0; includes NOTICE |
| scipy | BSD-3-Clause, with bundled numerical-library notices |
| setuptools, six, urllib3, wheel | MIT |
| typing_extensions | PSF-2.0 |
| ultralytics-thop | AGPL-3.0 |

Full license texts are distributed with the installed wheels under
`.venv/Lib/site-packages`, generally in each `*.dist-info` directory, its
`licenses` subdirectory, or the package itself. NumPy/SciPy metadata includes
substantial bundled-library license text, which is retained in the environment
report. Retain the original wheel notices when redistributing binaries; this
short inventory does not replace them. No third-party license is represented as
the license of VERTEBRATE itself.

P4B adds the test-only dependency **pytest-qt 4.5.0 (MIT)**. Its license is
retained in `.venv/Lib/site-packages/pytest_qt-4.5.0.dist-info/licenses/LICENSE`.
