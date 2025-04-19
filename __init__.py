import os
import importlib
import inspect
import re
import traceback

# Directory containing node wrapper modules relative to this __init__.py
WRAPPER_DIR = "node_wrappers"
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

# Get base path for the node_wrappers directory
base_dir_path = os.path.join(os.path.dirname(__file__), WRAPPER_DIR)

# Recursively scan all directories and subdirectories
def scan_for_modules(base_path, base_package):
    """Recursively scan directories for module files"""
    try:
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            
            # Skip __pycache__ and files/dirs starting with __
            if item.startswith('__'):
                continue
                
            if os.path.isdir(item_path):
                # Handle subdirectory
                subdir_package = f"{base_package}.{item}"
                
                # First load .py files in this directory
                try:
                    nodes, display_names = load_nodes_from_directory(subdir_package)
                    NODE_CLASS_MAPPINGS.update(nodes)
                    NODE_DISPLAY_NAME_MAPPINGS.update(display_names)
                except Exception as e:
                    print(f"\033[91m" + f"[ComfyUI_RealtimeNodes] Error processing directory {subdir_package}: {e}")
                    traceback.print_exc()
                
                # Then recursively process subdirectories
                subdir_path = os.path.join(base_path, item)
                scan_for_modules(subdir_path, subdir_package)
    except Exception as e:
        print(f"\033[91m" + f"[ComfyUI_RealtimeNodes] Error scanning directory {base_path}: {e}")
        traceback.print_exc()

# Start recursive scanning from the base directory
print(f"\033[92m" + f"[ComfyUI_RealtimeNodes] Scanning node_wrappers directory for modules...")
scan_for_modules(base_dir_path, WRAPPER_DIR)

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