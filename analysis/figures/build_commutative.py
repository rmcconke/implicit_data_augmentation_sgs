"""Assemble Fig. 4 (commutative.pdf) by swapping the raster panels of the figure.

Fig. 4 is a vector composite (arrows, labels, hatching, colour bars) that embeds 12
raster panels. This script opens an existing build of that composite (--template)
and replaces each embedded panel with the corresponding panel written by
analysis/figures/plot_commutative.py, so the layout stays untouched. The mapping from
embedded image names to panels below was found by exact pixel match against the
panels the template was built from; pass --check-panels with that panel directory to
re-verify it. The figure uses the non-augmented CNN panels. Requires pikepdf and
Pillow (pip install pikepdf pillow); CPU, seconds.

    python analysis/figures/build_commutative.py --template commutative_template.pdf \
        [--panels analysis/results/figures/commutative] [--out analysis/results/figures/commutative.pdf]
"""
import argparse
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--template', required=True, help='existing build of the Fig. 4 composite PDF')
parser.add_argument('--panels', default=os.path.join(_common.RESULTS, 'figures', 'commutative'))
parser.add_argument('--check-panels', default=None, help='panel directory the template was built from (optional check)')
parser.add_argument('--out', default=os.path.join(_common.RESULTS, 'figures', 'commutative.pdf'))
args = parser.parse_args()

import numpy as np  # noqa: E402
import pikepdf  # noqa: E402
from pikepdf import PdfImage, Name  # noqa: E402

MAP = {"/x19": "nearwall/x", "/x8": "nearwall/gx", "/x14": "nearwall/fx_noaug", "/x12": "nearwall/fgx_noaug",
       "/x13": "nearwall/gfx_noaug", "/x11": "nearwall/diff_noaug",
       "/x10": "middle/x", "/x9": "middle/gx", "/x18": "middle/fx_noaug", "/x16": "middle/fgx_noaug",
       "/x17": "middle/gfx_noaug", "/x15": "middle/diff_noaug"}
_open = []
def panel(path):
    p = pikepdf.open(path); _open.append(p)
    return list(p.pages[0].images.values())[0]
def cells(a, n=14):
    H, W, _ = a.shape
    ys = ((np.arange(n) + 0.5) * H / n).astype(int); xs = ((np.arange(n) + 0.5) * W / n).astype(int)
    return a[ys][:, xs]
pdf = pikepdf.open(args.template)
imgs = pdf.pages[0].images
for name, key in MAP.items():
    im = imgs[name]
    old = np.array(PdfImage(im).as_pil_image().convert("RGB"))
    H, W, _ = old.shape
    new_small = np.array(PdfImage(panel(os.path.join(args.panels, f"{key}.pdf"))).as_pil_image().convert("RGB"))
    if args.check_panels:
        # The template's raster must match the panel it was built from, to confirm the mapping.
        oldpanel = np.array(PdfImage(panel(os.path.join(args.check_panels, f"{key}.pdf"))).as_pil_image().convert("RGB"))
        assert np.abs(cells(old).astype(float) - cells(oldpanel)).max() == 0, name
    c = cells(new_small)                       # 14x14x3 new colours
    iy = np.minimum((np.arange(H) * 14) // H, 13); ix = np.minimum((np.arange(W) * 14) // W, 13)
    big = np.ascontiguousarray(c[iy][:, ix].astype(np.uint8))
    print(f"{name:5s} {key:22s} {W}x{H}  mean|new-old| per cell = {np.abs(c.astype(float) - cells(old)).mean():6.2f}")
    im.write(zlib.compress(big.tobytes()), filter=Name.FlateDecode)
    for k in ("/DecodeParms", "/SMask"):
        if k in im: del im[k]
pdf.save(args.out)
print("saved", args.out)
