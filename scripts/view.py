"""Open the interactive MuJoCo window with the scene (drag to orbit, double-click a body to select).
Needs a display, so run it on your own machine, not headless.
 
    python scripts/view.py
"""
import _common as c
import mujoco
import mujoco.viewer
 
from ur3pick.pipeline import load_cfg
from ur3pick.sim.scene import Scene
 
scene = Scene(load_cfg())
scene.reset(0)
mujoco.viewer.launch(scene.model, scene.data)