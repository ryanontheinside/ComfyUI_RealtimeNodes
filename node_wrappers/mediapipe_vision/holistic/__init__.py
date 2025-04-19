"""MediaPipe Holistic node wrappers.

this module contains the nodes for the holistic solution. Holistic is only available in the legacy api
and thus is stands aout among the other imeplementations in this project. When the official update
to this is released, this module will be updated to use the new api.
"""

# Import modules to ensure they're registered
from . import holistic_landmark
from . import holistic_landmark_visualization 