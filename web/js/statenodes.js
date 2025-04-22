import { app } from "../../../scripts/app.js";
import { stateApi } from "./stateApi.js";

// Workflow namespace manager
const workflowManager = {
    // Get the current workflow ID or generate a new one
    getWorkflowId: function() {
        if (!app.graph.extra) {
            app.graph.extra = {};
        }
        
        if (!app.graph.extra.realtimeNodesState) {
            // Generate a new workflow ID (timestamp + random)
            const newId = `wf_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
            app.graph.extra.realtimeNodesState = { workflowId: newId };
            app.graph.change(); // Ensure it gets saved
        }
        
        return app.graph.extra.realtimeNodesState.workflowId;
    },
    
    // Set a state value using the API
    setState: function(key, value) {
        const workflowId = this.getWorkflowId();
        return stateApi.setValue(workflowId, key, value);
    },
    
    // Get a state value using the API
    getState: function(key) {
        const workflowId = this.getWorkflowId();
        return stateApi.getValue(workflowId, key);
    },
    
    // Delete a state value using the API
    deleteState: function(key) {
        const workflowId = this.getWorkflowId();
        return stateApi.deleteValue(workflowId, key);
    },
    
    // Get all state values for the current workflow
    getAllState: function() {
        const workflowId = this.getWorkflowId();
        return stateApi.getAllValues(workflowId);
    },
    
    // Clear all state for the current workflow
    clearAllState: function() {
        const workflowId = this.getWorkflowId();
        return stateApi.clearAllValues(workflowId);
    },
    
    // Create a namespaced key using the workflow ID (for display only)
    getNamespacedKey: function(key) {
        return `${this.getWorkflowId()}:${key}`;
    },
    
    // Extract the user-facing key from a namespaced key
    extractUserKey: function(namespacedKey) {
        if (!namespacedKey) return "";
        const parts = namespacedKey.split(":");
        return parts.length > 1 ? parts[1] : namespacedKey;
    }
};

// Track all keys available
const stateKeys = new Set();

// Track connections to visualize data flow between set and get nodes
const connectionVisualizer = {
    // Map of key -> {setNode, getNodes}
    connections: {},
    
    registerSetNode: function(node, key) {
        if (!key) return;
        
        // Initialize if not exists
        if (!this.connections[key]) {
            this.connections[key] = { setNode: null, getNodes: [] };
        }
        
        this.connections[key].setNode = node;
        
        // Find corresponding get nodes
        this.updateConnections(key);
    },
    
    registerGetNode: function(node, key) {
        if (!key) return;
        
        // Initialize if not exists
        if (!this.connections[key]) {
            this.connections[key] = { setNode: null, getNodes: [] };
        }
        
        // Add to getNodes if not already there
        if (!this.connections[key].getNodes.includes(node)) {
            this.connections[key].getNodes.push(node);
        }
        
        // Update connections
        this.updateConnections(key);
    },
    
    unregisterNode: function(node, key) {
        if (!key || !this.connections[key]) return;
        
        // Remove from connections
        if (this.connections[key].setNode === node) {
            this.connections[key].setNode = null;
        }
        
        const getNodes = this.connections[key].getNodes;
        const index = getNodes.indexOf(node);
        if (index !== -1) {
            getNodes.splice(index, 1);
        }
        
        // Remove key if no nodes left
        if (!this.connections[key].setNode && this.connections[key].getNodes.length === 0) {
            delete this.connections[key];
        }
    },
    
    updateConnections: function(key) {
        if (!key || !this.connections[key]) return;
        
        const connection = this.connections[key];
        
        // Update visual properties for set node
        if (connection.setNode) {
            connection.setNode.title = `Set State: ${key}`;
        }
        
        // Update visual properties for get nodes
        connection.getNodes.forEach(getNode => {
            getNode.title = `Get State: ${key}`;
        });
    },
    
    getSetNodeForKey: function(key) {
        return this.connections[key]?.setNode;
    },
    
    getNodesForKey: function(key) {
        return this.connections[key]?.getNodes || [];
    }
};

// Helper to show alerts to the user
function showAlert(message) {
  app.extensionManager.toast.add({
    severity: 'info',
    summary: "State Nodes",
    detail: `${message}`,
    life: 3000,
  });
}

// Helper to find all set nodes with a given key
function findSetNodesWithKey(graph, key) {
    return graph._nodes.filter(node => 
        node.type === "SetStateNode" && 
        node.widgets && 
        node.widgets.find(w => w.name === "key" && w.value === key)
    );
}

// Helper to find all get nodes with a given key
function findGetNodesWithKey(graph, key) {
    return graph._nodes.filter(node => 
        node.type === "GetStateNode" && 
        node.widgets && 
        node.widgets.find(w => w.name === "key" && w.value === key)
    );
}

// Helper to find all get nodes in the graph
function findAllGetNodes(graph) {
    return graph._nodes.filter(node => node.type === "GetStateNode");
}

// Helper to update all get nodes with the current list of keys
function updateAllGetNodesDropdowns(graph) {
    const getNodes = findAllGetNodes(graph);
    getNodes.forEach(getNode => {
        const keyWidget = getNode.widgets.find(w => w.name === "key");
        if (keyWidget && keyWidget.options) {
            // Update the values in the dropdown
            keyWidget.options.values = [...stateKeys].sort();
            
            // If the current key no longer exists, reset it
            if (keyWidget.value && !stateKeys.has(keyWidget.value)) {
                keyWidget.value = "";
                getNode.title = "Get State";
            }
        }
    });
}

// Helper to properly disable a widget
function disableWidget(widget) {
    if (widget) {
        widget.disabled = true;
        widget.visibleWidth = 0;
        widget.onMouseDown = () => {};
        widget.onMouseMove = () => {};
        widget.onMouseUp = () => {};
    }
}

// Register the SetStateNode for UI customization
app.registerExtension({
    name: "SetStateNode",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SetStateNode") return;
        
        // Add custom widgets and behavior
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            // Store reference to the node
            const node = this;
            
            // Add hidden workflow_id widget
            // Find if workflow_id widget already exists
            let workflowIdWidget = node.widgets.find(w => w.name === "workflow_id");
            if (!workflowIdWidget) {
                const workflowId = workflowManager.getWorkflowId();
                workflowIdWidget = node.addWidget("text", "workflow_id", workflowId, () => {}, {
                    serialize: false,
                    disabled: true
                });
                workflowIdWidget.hidden = true;
                disableWidget(workflowIdWidget);
            }
            
            // Track key changes
            const keyWidget = node.widgets.find(w => w.name === "key");
            if (keyWidget) {
                const originalCallback = keyWidget.callback;
                keyWidget.callback = function(value) {
                    if (originalCallback) {
                        originalCallback.apply(this, arguments);
                    }
                    
                    // Add to available keys
                    if (value && value.trim() !== "") {
                        stateKeys.add(value);
                        node.title = "Set State: " + value;
                        
                        // Store the namespaced key in a hidden property
                        node._namespacedKey = workflowManager.getNamespacedKey(value);
                        
                        // Update all GetStateNodes
                        if (node.graph) {
                            updateAllGetNodesDropdowns(node.graph);
                        }
                    }
                };
                
                // Initialize if there's a default value
                if (keyWidget.value && keyWidget.value.trim() !== "") {
                    stateKeys.add(keyWidget.value);
                    node.title = "Set State: " + keyWidget.value;
                    node._namespacedKey = workflowManager.getNamespacedKey(keyWidget.value);
                }
            }
            
            // Set node appearance
            this.color = "#2a363b";
            this.bgcolor = "#3f5159";
            
            // Add extra context menu items
            const getExtraMenuOptions = node.getExtraMenuOptions;
            node.getExtraMenuOptions = function(_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }
                
                const keyWidget = node.widgets.find(w => w.name === "key");
                if (keyWidget && keyWidget.value) {
                    const key = keyWidget.value;
                    
                    // Add menu to see all connected get nodes
                    const getNodes = findGetNodesWithKey(this.graph, key);
                    if (getNodes.length > 0) {
                        const getNodesSubmenu = getNodes.map(getNode => ({
                            content: `Get Node: ${getNode.id}`,
                            callback: () => {
                                app.canvas.centerOnNode(getNode);
                                app.canvas.selectNode(getNode);
                            }
                        }));
                        
                        options.unshift({
                            content: "Connected Get Nodes",
                            has_submenu: true,
                            submenu: {
                                options: getNodesSubmenu
                            }
                        });
                    }
                }
            };
            
            // When the node is removed
            const onRemoved = node.onRemoved;
            node.onRemoved = function() {
                if (onRemoved) {
                    onRemoved.apply(this, arguments);
                }
                
                // Check if we need to remove the key from the available keys
                const keyWidget = node.widgets.find(w => w.name === "key");
                if (keyWidget && keyWidget.value) {
                    const key = keyWidget.value;
                    // Only remove if no other SetStateNodes are using this key
                    const otherSetNodes = findSetNodesWithKey(this.graph, key);
                    if (otherSetNodes.length <= 1) { // 1 because this node is still in the list
                        stateKeys.delete(key);
                        
                        // Update the dropdowns in all GetStateNodes
                        updateAllGetNodesDropdowns(this.graph);
                    }
                }
            };
            
            // Hook into the onExecuted event to ensure we're using the namespaced key
            const onExecuted = node.onExecuted;
            node.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                // Get key and value from the node
                const keyWidget = node.widgets.find(w => w.name === "key");
                const valueWidget = node.widgets.find(w => w.name === "value");
                
                if (keyWidget && keyWidget.value && keyWidget.value.trim() !== "") {
                    // Save the value to the server-side state
                    const key = keyWidget.value;
                    const value = message?.output?.value; // Get value from execution result
                    
                    // Use the state API to save the value
                    workflowManager.setState(key, value)
                        .then(() => {
                            console.log(`State value set for key: ${key}`);
                        })
                        .catch(error => {
                            console.error(`Error setting state value for key: ${key}`, error);
                        });
                }
            };
        };
        
        // Modify the serialize method to include the namespaced key
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(info) {
            if (onSerialize) {
                onSerialize.apply(this, arguments);
            }
            
            // Add the namespaced key to the serialized data
            const keyWidget = this.widgets.find(w => w.name === "key");
            if (keyWidget && keyWidget.value && keyWidget.value.trim() !== "") {
                info.namespacedKey = workflowManager.getNamespacedKey(keyWidget.value);
            }
        };
        
        // Modify the configure method to restore the namespaced key
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            
            // Restore the namespaced key from the serialized data
            if (info.namespacedKey) {
                this.properties.namespacedKey = info.namespacedKey;
            } else {
                const keyWidget = this.widgets.find(w => w.name === "key");
                if (keyWidget && keyWidget.value && keyWidget.value.trim() !== "") {
                    this.properties.namespacedKey = workflowManager.getNamespacedKey(keyWidget.value);
                }
            }
        };
    }
});

// Register the GetStateNode for UI customization
app.registerExtension({
    name: "GetStateNode",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GetStateNode") return;
        
        // Add custom widgets and behavior
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            // Store reference to the node
            const node = this;
            
            // Add hidden workflow_id widget
            // Find if workflow_id widget already exists
            let workflowIdWidget = node.widgets.find(w => w.name === "workflow_id");
            if (!workflowIdWidget) {
                const workflowId = workflowManager.getWorkflowId();
                workflowIdWidget = node.addWidget("text", "workflow_id", workflowId, () => {}, {
                    serialize: false,
                    disabled: true
                });
                workflowIdWidget.hidden = true;
                disableWidget(workflowIdWidget);
            }
            
            // Set node appearance
            this.color = "#2a363b";
            this.bgcolor = "#3f5159";
            
            // Find the key widget and enhance it
            const keyWidget = node.widgets.find(w => w.name === "key");
            if (keyWidget) {
                // Replace widget with combo
                const index = node.widgets.indexOf(keyWidget);
                const keyValue = keyWidget.value; // Save current value
                
                // Remove old widget
                node.widgets.splice(index, 1);
                
                // Add new combo widget
                const newWidget = node.addWidget("combo", "key", keyValue, function(value) {
                    if (value) {
                        node.title = "Get State: " + value;
                        // Store the namespaced key in a hidden property
                        node._namespacedKey = workflowManager.getNamespacedKey(value);
                    } else {
                        node.title = "Get State";
                        node._namespacedKey = "";
                    }
                }, { 
                    values: () => [...stateKeys].sort(),
                    name: "key", // Ensure name matches for server
                });
                
                // Move it to the correct position
                if (index !== node.widgets.length - 1) {
                    const removed = node.widgets.pop();
                    node.widgets.splice(index, 0, removed);
                }
                
                // Initialize title and namespaced key
                if (newWidget.value) {
                    node.title = "Get State: " + newWidget.value;
                    node._namespacedKey = workflowManager.getNamespacedKey(newWidget.value);
                } else {
                    node.title = "Get State";
                }
            }
            
            // Add extra context menu items
            const getExtraMenuOptions = node.getExtraMenuOptions;
            node.getExtraMenuOptions = function(_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }
                
                const keyWidget = node.widgets.find(w => w.name === "key");
                if (keyWidget && keyWidget.value) {
                    const key = keyWidget.value;
                    
                    // Add menu to go to set nodes with this key
                    const setNodes = findSetNodesWithKey(this.graph, key);
                    if (setNodes.length > 0) {
                        const setNodesSubmenu = setNodes.map(setNode => ({
                            content: `Set Node: ${setNode.id}`,
                            callback: () => {
                                app.canvas.centerOnNode(setNode);
                                app.canvas.selectNode(setNode);
                            }
                        }));
                        
                        options.unshift({
                            content: setNodes.length === 1 ? "Go to Set Node" : "Go to Set Nodes",
                            has_submenu: setNodes.length > 1,
                            callback: setNodes.length === 1 ? (() => {
                                app.canvas.centerOnNode(setNodes[0]);
                                app.canvas.selectNode(setNodes[0]);
                            }) : undefined,
                            submenu: setNodes.length > 1 ? {
                                options: setNodesSubmenu
                            } : undefined
                        });
                    }
                    
                    // Add menu to see other get nodes with the same key
                    const getNodes = findGetNodesWithKey(this.graph, key).filter(n => n !== this);
                    if (getNodes.length > 0) {
                        const getNodesSubmenu = getNodes.map(getNode => ({
                            content: `Get Node: ${getNode.id}`,
                            callback: () => {
                                app.canvas.centerOnNode(getNode);
                                app.canvas.selectNode(getNode);
                            }
                        }));
                        
                        options.unshift({
                            content: "Other Get Nodes",
                            has_submenu: true,
                            submenu: {
                                options: getNodesSubmenu
                            }
                        });
                    }
                }
            };
            
            // Hook into the onExecuted event to ensure we're using the namespaced key
            const onExecuted = node.onExecuted;
            node.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                // Get key from the node
                const keyWidget = node.widgets.find(w => w.name === "key");
                
                if (keyWidget && keyWidget.value && keyWidget.value.trim() !== "") {
                    const key = keyWidget.value;
                    
                    // Check if we need to refresh the state (for debugging)
                    if (message?.output !== undefined) {
                        // We could fetch the current value for validation
                        workflowManager.getState(key)
                            .then(value => {
                                console.log(`Retrieved state for key: ${key}`, value);
                            })
                            .catch(error => {
                                console.error(`Error retrieving state for key: ${key}`, error);
                            });
                    }
                }
            };
        };
        
        // Modify the serialize method to include the namespaced key
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(info) {
            if (onSerialize) {
                onSerialize.apply(this, arguments);
            }
            
            // Add the namespaced key to the serialized data
            const keyWidget = this.widgets.find(w => w.name === "key");
            if (keyWidget && keyWidget.value && keyWidget.value.trim() !== "") {
                info.namespacedKey = workflowManager.getNamespacedKey(keyWidget.value);
            }
        };
        
        // Modify the configure method to restore the namespaced key
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            
            // Restore the namespaced key from the serialized data
            if (info.namespacedKey) {
                this.properties.namespacedKey = info.namespacedKey;
            } else {
                const keyWidget = this.widgets.find(w => w.name === "key");
                if (keyWidget && keyWidget.value && keyWidget.value.trim() !== "") {
                    this.properties.namespacedKey = workflowManager.getNamespacedKey(keyWidget.value);
                }
            }
        };
    }
});

// Register the other state-related nodes
app.registerExtension({
    name: "StateUtilityNodes",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ClearStateNode" && nodeData.name !== "ListStatesNode") return;
        
        // Add custom appearance
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            // Set node appearance
            this.color = "#2a363b";
            this.bgcolor = "#3f5159";
            
            // When ClearStateNode is executed, also update the UI
            if (nodeData.name === "ClearStateNode") {
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    if (onExecuted) {
                        onExecuted.apply(this, arguments);
                    }
                    
                    // Check if all states were cleared
                    const action = this.widgets.find(w => w.name === "action");
                    if (action && action.value === "Clear All States") {
                        // Clear the state keys set
                        stateKeys.clear();
                        showAlert("All states cleared");
                        
                        // Update all GetStateNodes
                        updateAllGetNodesDropdowns(this.graph);
                    } else {
                        // Clear just one key
                        const keyWidget = this.widgets.find(w => w.name === "key");
                        if (keyWidget && keyWidget.value) {
                            stateKeys.delete(keyWidget.value);
                            showAlert(`State '${keyWidget.value}' cleared`);
                            
                            // Update all GetStateNodes
                            updateAllGetNodesDropdowns(this.graph);
                        }
                    }
                };
            }
        };
    }
});

// Initialize workflow ID when graph is loaded or created
app.registerExtension({
    name: "RealtimeNodesWorkflowID",
    async setup() {
        // Initialize workflow ID on startup
        workflowManager.getWorkflowId();
        
        // Helper function to update workflow_id in all state nodes
        const updateWorkflowIdInNodes = () => {
            const workflowId = workflowManager.getWorkflowId();
            
            // Update all SetStateNodes
            app.graph._nodes.forEach(node => {
                if (node.type === "SetStateNode" || node.type === "GetStateNode") {
                    const workflowIdWidget = node.widgets.find(w => w.name === "workflow_id");
                    if (workflowIdWidget) {
                        workflowIdWidget.value = workflowId;
                        disableWidget(workflowIdWidget);
                    } else {
                        // Add the widget if it doesn't exist
                        const newWidget = node.addWidget("text", "workflow_id", workflowId, () => {}, {
                            serialize: false,
                            disabled: true
                        });
                        newWidget.hidden = true;
                        disableWidget(newWidget);
                    }
                }
            });
        };
        
        // Register graph observer to update nodes when graph changes
        app.graph.onNodeAdded = function(node) {
            if (node.type === "SetStateNode" || node.type === "GetStateNode") {
                // Let the node creation process complete first
                setTimeout(() => {
                    const workflowId = workflowManager.getWorkflowId();
                    const workflowIdWidget = node.widgets.find(w => w.name === "workflow_id");
                    if (workflowIdWidget) {
                        workflowIdWidget.value = workflowId;
                        disableWidget(workflowIdWidget);
                    }
                }, 50);
            }
        };
        
        // Hook into graph serialization to ensure workflow ID is set
        const originalGraphToJSON = app.graph.toJSON;
        app.graph.toJSON = function(data) {
            // Ensure workflow ID is up to date before serialization
            updateWorkflowIdInNodes();
            
            // Call original method
            return originalGraphToJSON.call(this, data);
        };
    },
    async beforeClearGraph() {
        // Generate a new ID when graph is cleared
        if (app.graph.extra && app.graph.extra.realtimeNodesState) {
            // Generate a new workflow ID for the new graph
            const newId = `wf_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
            app.graph.extra.realtimeNodesState.workflowId = newId;
        }
    },
    async graphLoaded() {
        // Make sure the workflow ID is established
        // Use a small delay to ensure the graph is fully loaded
        setTimeout(() => {
            const workflowId = workflowManager.getWorkflowId();
            
            // Update all state nodes with the current workflow ID
            app.graph._nodes.forEach(node => {
                if (node.type === "SetStateNode" || node.type === "GetStateNode") {
                    // Find or add the workflow_id widget
                    let workflowIdWidget = node.widgets.find(w => w.name === "workflow_id");
                    if (workflowIdWidget) {
                        workflowIdWidget.value = workflowId;
                    } else {
                        // Add the widget if it doesn't exist
                        const newWidget = node.addWidget("text", "workflow_id", workflowId, () => {}, {
                            serialize: false,
                            disabled: true
                        });
                        newWidget.hidden = true;
                        disableWidget(newWidget);
                    }
                }
            });
        }, 100);
    }
}); 