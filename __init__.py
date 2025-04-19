import os
import importlib
import inspect
import re
import traceback

# Directory containing node wrapper modules relative to this __init__.py
WRAPPER_DIR = "node_wrappers.realtimenodes"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("\033[93m" + "[ComfyUI_RealtimeNodes] Loading nodes...")

def load_nodes_from_directory(package_path):
    """Dynamically loads nodes from Python files in a directory."""
    loaded_nodes = {}
    loaded_display_names = {}
    
    # Calculate absolute path to the package directory
    package_abs_path = os.path.join(os.path.dirname(__file__), *package_path.split('.'))
    
    if not os.path.isdir(package_abs_path):
        print(f"\033[91m" + f"[ComfyUI_RealtimeNodes] Warning: Directory not found: {package_abs_path}")
        return loaded_nodes, loaded_display_names

    print(f"\033[92m" + f"[ComfyUI_RealtimeNodes] Searching for nodes in: {package_abs_path}")

    for filename in os.listdir(package_abs_path):
        if filename.endswith(".py") and not filename.startswith('__'):
            module_name = filename[:-3]
            full_module_path = f".{package_path}.{module_name}" # Relative import path
            
            try:
                module = importlib.import_module(full_module_path, package=__name__)
                
                # Find classes defined in the module
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj):
                        # Check if it's a ComfyUI node (heuristic: has INPUT_TYPES)
                        if hasattr(obj, 'INPUT_TYPES') and callable(obj.INPUT_TYPES):
                             # Exclude base classes/mixins if they are explicitly named or lack a CATEGORY
                             if not name.endswith("Base") and not name.endswith("Mixin") and hasattr(obj, 'CATEGORY'):
                                loaded_nodes[name] = obj
                                
                                # Generate display name (similar to original logic)
                                display_name = ' '.join(word.capitalize() for word in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name))
                                suffix = " 🕒🅡🅣🅝"
                                if not display_name.endswith(suffix):
                                    display_name += suffix
                                loaded_display_names[name] = display_name
                                print(f"\033[94m" + f"[ComfyUI_RealtimeNodes]   - Loaded node: {name} -> {display_name}")
                                
            except ImportError as e:
                print(f"\033[91m" + f"[ComfyUI_RealtimeNodes] Error importing module {full_module_path}: {e}")
                traceback.print_exc()
            except Exception as e:
                print(f"\033[91m" + f"[ComfyUI_RealtimeNodes] Error processing module {full_module_path}: {e}")
                traceback.print_exc()
                
    return loaded_nodes, loaded_display_names

# --- Main Loading Logic ---

# Define the subdirectories within WRAPPER_DIR to scan
# Order might matter if there are dependencies between modules, though ideally there shouldn't be
subdirs_to_scan = [
    "controls",
    "media",
    "utils",
    "specialized" # Add other subdirs like 'specialized' if needed
]

# Iterate through specified subdirectories and load nodes
for subdir in subdirs_to_scan:
    dir_path = f"{WRAPPER_DIR}.{subdir}"
    nodes, display_names = load_nodes_from_directory(dir_path)
    NODE_CLASS_MAPPINGS.update(nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(display_names)


suffix = " 🕒🅡🅣🅝"

for node_name in NODE_CLASS_MAPPINGS.keys():
    # Convert camelCase or snake_case to Title Case
    if node_name not in NODE_DISPLAY_NAME_MAPPINGS:
        display_name = ' '.join(word.capitalize() for word in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', node_name))
    else:
        display_name = NODE_DISPLAY_NAME_MAPPINGS[node_name]
    
    # Add the suffix if it's not already present
    if not display_name.endswith(suffix):
        display_name += suffix
    
    # Assign the final display name to the mappings
    NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name
# --- Original Web Directory and Export ---
WEB_DIRECTORY = "./web/js" # Adjusted path if needed

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("\033[92m" + f"[ComfyUI_RealtimeNodes] Loaded {len(NODE_CLASS_MAPPINGS)} nodes.") 