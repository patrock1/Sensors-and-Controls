"""test_detection.py"""

import numpy as np
from ur3pick.sensing.detection import detect_object

def _patch(rgb_colour):
    img = np.full((60, 60, 3), 128, np.uint8)
    img[20:40, 20:40] = rgb_colour
    return img

def test_detects_red_only(cfg):
    """The hue wrap must be 170: a wrong value makes every non-red colour look red."""
    d = cfg["detection"]
    assert detect_object(_patch((230, 20, 20)), d).found
    assert not detect_object(_patch((0, 200, 0)), d).found       # green
    assert not detect_object(_patch((0, 0, 255)), d).found       # blue
    assert not detect_object(_patch((230, 200, 20)), d).found    # yellow
    assert not detect_object(_patch((128, 128, 128)), d).found   # grey table