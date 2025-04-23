import { app } from "../../../scripts/app.js";

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
                    } else {
                        node.title = "Get State";
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
                
                // Initialize title
                if (newWidget.value) {
                    node.title = "Get State: " + newWidget.value;
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