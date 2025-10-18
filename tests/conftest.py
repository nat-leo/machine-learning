import sys
import os

# Add the project's src directory to the sys.path to ensure the ml package is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
