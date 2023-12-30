
# Interactive Geometry Remeshing

A Python implementation of Pierre Alliez, Mark Meyer, and Mathieu Desbrun's research paper "Interactive Geometry Remeshing."

Link to google drive with demo video: https://drive.google.com/drive/folders/1yvPP3hPceYbScddZL6QRBsEnUGWdBCpy?usp=drive_link


## Dependencies:

All geometry processing is done with NumPy and igl, Meshplot is used for plotting, and I did have to use VisPy and Pillow (PIL)or handling some image processing, although neither are used for any built-in mesh/geometry algorithms. 
In order to run VisPy in a Jupyter notebook you also need Jupyter_rfb installed in your environment, although I don't think it needs to be explicitly imported in the notebook.



Full list of imports:
```
import math
import random
import os
import igl
import numpy as np
import meshplot as mp
import scipy as sp
import ipywidgets as iw
import vispy
from vispy import io
from vispy import app, gloo
from vispy.gloo import Program
from PIL import Image
```

