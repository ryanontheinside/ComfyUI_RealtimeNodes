// Widget validation for RealTime nodes
import { app } from "../../../scripts/app.js";

// Register validation behavior when nodes are connected
app.registerExtension({
    name: "RealTime.WidgetValidation",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Skip if not a real-time control node
        if (!nodeData.category?.startsWith("real-time/control/value")) {
            return;
        }
        
        console.log("Registering validation for control node:", nodeData.name);

        // Store constraints from connected widgets
        nodeType.prototype.targetConstraints = null;

        // Add handler for when node is created (including on page load)
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            
            // Use setTimeout to check connections after graph is loaded
            setTimeout(() => {
                // Check existing connections on load
                const outputLinks = this.outputs[0]?.links || [];
                console.log("Checking existing connections on load (delayed):", outputLinks);
                
                for (const linkId of outputLinks) {
                    const link = app.graph.links[linkId];
                    if (!link) continue;

                    const targetNode = app.graph.getNodeById(link.target_id);
                    const targetSlot = link.target_slot;
                    
                    if (targetNode?.widgets) {
                        const inputName = targetNode.inputs[targetSlot]?.name;
                        const targetWidget = targetNode.widgets.find(w => w.name === inputName);
                        
                        if (targetWidget?.options) {
                            console.log("Found existing connection to widget:", targetWidget);
                            this.targetConstraints = {
                                min: targetWidget.options.min,
                                max: targetWidget.options.max,
                                step: targetWidget.options.step
                            };
                            this.updateWidgetConstraints();
                        }
                    }
                }
            }, 100); // Small delay to ensure graph is loaded
            
            return result;
        };

        const originalOnConnectOutput = nodeType.prototype.onConnectOutput;
        nodeType.prototype.onConnectOutput = function (slot, type, input, targetNode, targetSlot) {
            console.log("Output connection made from control node:", this.title);
            console.log("Target node:", targetNode?.title);
            console.log("Target slot:", targetSlot);
            console.log("All target widgets:", targetNode?.widgets);
            
            // Call original connection handler
            const result = originalOnConnectOutput?.apply(this, arguments);

            if (targetNode?.widgets) {
                // Find widget by input name instead of slot index
                const inputName = targetNode.inputs[targetSlot]?.name;
                console.log("Looking for widget matching input name:", inputName);
                
                const targetWidget = targetNode.widgets.find(w => w.name === inputName);
                console.log("Found target widget:", targetWidget);
                
                if (targetWidget?.options) {
                    console.log("Target widget options:", targetWidget.options);
                    
                    // Store the constraints including step
                    this.targetConstraints = {
                        min: targetWidget.options.min,
                        max: targetWidget.options.max,
                        step: targetWidget.options.step
                    };

                    console.log("Setting constraints:", this.targetConstraints);
                    
                    // Update widgets with new constraints
                    const widgets = ["maximum_value", "minimum_value", "starting_value"];
                    widgets.forEach(name => {
                        const widget = this.widgets.find(w => w.name === name);
                        if (widget) {
                            if (this.targetConstraints.min !== undefined) {
                                widget.options.min = this.targetConstraints.min;
                            }
                            if (this.targetConstraints.max !== undefined) {
                                widget.options.max = this.targetConstraints.max;
                            }
                            if (this.targetConstraints.step !== undefined) {
                                widget.options.step = this.targetConstraints.step;
                            }
                        }
                    });

                    // Clamp current values
                    this.clampWidgetValues();
                } else {
                    console.log("Widget has no options");
                }
            }

            return result;
        };

        // Add method to update widget constraints
        nodeType.prototype.updateWidgetConstraints = function() {
            if (!this.targetConstraints) return;

            const widgets = ["maximum_value", "minimum_value", "starting_value"];
            widgets.forEach(name => {
                const widget = this.widgets.find(w => w.name === name);
                if (widget) {
                    if (this.targetConstraints.min !== undefined) {
                        widget.options.min = this.targetConstraints.min;
                    }
                    if (this.targetConstraints.max !== undefined) {
                        widget.options.max = this.targetConstraints.max;
                    }
                }
            });

            // Clamp current values to new constraints
            this.clampWidgetValues();
        };

        // Add method to clamp widget values
        nodeType.prototype.clampWidgetValues = function() {
            const minWidget = this.widgets.find(w => w.name === "minimum_value");
            const maxWidget = this.widgets.find(w => w.name === "maximum_value");
            const startWidget = this.widgets.find(w => w.name === "starting_value");

            if (minWidget && maxWidget) {
                // Ensure min <= start <= max and all within target constraints
                const targetMin = this.targetConstraints?.min ?? minWidget.options.min;
                const targetMax = this.targetConstraints?.max ?? maxWidget.options.max;

                minWidget.value = Math.max(targetMin, Math.min(minWidget.value, maxWidget.value));
                maxWidget.value = Math.max(minWidget.value, Math.min(maxWidget.value, targetMax));
                
                if (startWidget) {
                    startWidget.value = Math.max(minWidget.value, Math.min(startWidget.value, maxWidget.value));
                }
            }
        };

        // Override the widget's callback to enforce constraints
        const originalWidgetCallback = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function(widget, value) {
            const result = originalWidgetCallback?.apply(this, arguments);
            
            // If this is one of our constrained widgets, enforce the constraints
            if (["maximum_value", "minimum_value", "starting_value"].includes(widget.name)) {
                this.clampWidgetValues();
            }
            
            return result;
        };
    }
}); 