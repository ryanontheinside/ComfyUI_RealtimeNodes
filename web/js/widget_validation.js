// Widget validation for RealTime nodes
import { app } from "../../../scripts/app.js";

// Register validation behavior when nodes are connected
app.registerExtension({
    name: "RealTime.WidgetValidation",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Update category check to include sequence nodes
        if (!nodeData.category?.startsWith("real-time/control/")) {
            return;
        }
        
        console.log("Registering validation for control node:", nodeData.name);

        // Store constraints from connected widgets
        nodeType.prototype.targetConstraints = null;

        // Add handler for when node is created (including on page load)
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            
            // Wrap the values widget callback to ensure validation
            const valuesWidget = this.widgets.find(w => w.name === "values");
            if (valuesWidget) {
                const originalCallback = valuesWidget.callback;
                valuesWidget.callback = (value) => {
                    console.log("Values widget direct callback:", {
                        before: value,
                        hasConstraints: !!this.targetConstraints
                    });
                    
                    if (this.targetConstraints) {
                        this.validateSequenceValues(valuesWidget);
                        value = valuesWidget.value;  // Use validated value
                    }
                    
                    console.log("After validation:", {
                        after: value
                    });
                    
                    return originalCallback?.call(valuesWidget, value);
                };
            }
            
            // Function to check connections
            const checkConnections = () => {
                const outputLinks = this.outputs[0]?.links || [];
                console.log("Checking existing connections:", outputLinks);
                
                for (const linkId of outputLinks) {
                    const link = app.graph.links[linkId];
                    if (!link) continue;

                    const targetNode = app.graph.getNodeById(link.target_id);
                    const targetSlot = link.target_slot;
                    
                    if (targetNode?.widgets) {
                        const inputName = targetNode.inputs[targetSlot]?.name;
                        const targetWidget = targetNode.widgets.find(w => w.name === inputName);
                        
                        if (targetWidget?.options || (targetWidget?.type === "converted-widget" && targetWidget.options)) {
                            console.log("Found existing connection to widget:", targetWidget);
                            this.targetConstraints = {
                                min: targetWidget.options.min,
                                max: targetWidget.options.max,
                                step: targetWidget.options.step
                            };

                            // For sequence nodes, validate immediately
                            if (nodeData.category?.includes("/sequence")) {
                                const valuesWidget = this.widgets.find(w => w.name === "values");
                                if (valuesWidget) {
                                    this.validateSequenceValues(valuesWidget);
                                }
                            } else {
                                this.updateWidgetConstraints();
                            }
                        }
                    }
                }
            };

            // Check immediately
            checkConnections();
            
            // Also check after a delay to ensure graph is fully loaded
            setTimeout(checkConnections, 100);
            
            // And check once more after a longer delay
            setTimeout(checkConnections, 1000);
            
            return result;
        };

        const originalOnConnectOutput = nodeType.prototype.onConnectOutput;
        nodeType.prototype.onConnectOutput = function (slot, type, input, targetNode, targetSlot) {
            console.log("Output connection made from control node:", this.title);
            
            // Call original connection handler if it exists
            const result = originalOnConnectOutput?.apply(this, arguments);

            if (targetNode?.widgets) {
                const inputName = targetNode.inputs[targetSlot]?.name;
                const targetWidget = targetNode.widgets.find(w => w.name === inputName);
                
                console.log("Found target widget:", targetWidget);
                
                if (targetWidget?.options || (targetWidget?.type === "converted-widget" && targetWidget.options)) {
                    // Handle both regular and converted widgets
                    const options = targetWidget.options;
                    this.targetConstraints = {
                        min: options.min,
                        max: options.max,
                        step: options.step
                    };

                    console.log("Set constraints:", this.targetConstraints);

                    // For sequence nodes, validate the values string
                    if (nodeData.category?.includes("/sequence")) {
                        const valuesWidget = this.widgets.find(w => w.name === "values");
                        if (valuesWidget) {
                            this.validateSequenceValues(valuesWidget);
                        }
                    } else {
                        // Existing validation for non-sequence nodes
                        this.updateWidgetConstraints();
                    }
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

        // Add method to validate sequence values
        nodeType.prototype.validateSequenceValues = function(widget) {
            if (!this.targetConstraints) {
                console.log("No target constraints found for validation");
                return;
            }

            console.log("Validating sequence values:", {
                widget: widget,
                value: widget.value,
                constraints: this.targetConstraints
            });

            const separator = widget.options?.multiline ? "\n" : ",";
            const values = widget.value.split(separator)
                .map(v => v.trim())
                .filter(v => v.length > 0);

            // Convert and validate values based on node type
            const isFloat = this.type === "FloatSequence";
            const isInt = this.type === "IntSequence";

            console.log("Processing values:", {
                nodeType: this.type,
                isFloat: isFloat,
                isInt: isInt,
                values: values
            });

            if (isFloat || isInt) {
                const validValues = values.map(v => {
                    let num = isFloat ? parseFloat(v) : parseInt(v);
                    if (isNaN(num)) num = this.targetConstraints.min;
                    const clamped = Math.max(this.targetConstraints.min, Math.min(num, this.targetConstraints.max));
                    console.log(`Validating value: ${v} -> ${num} -> ${clamped}`);
                    return isFloat ? clamped.toString() : Math.round(clamped).toString();
                });

                // Update the widget value with validated values
                const newValue = validValues.join(separator);
                console.log("Setting new value:", newValue);
                widget.value = newValue;
            }
        };

        // Override the widget's callback to enforce constraints
        const originalWidgetCallback = nodeType.prototype.onWidgetChanged;
        console.log("Setting up widget change handler for:", nodeData.name, {
            hasOriginalCallback: !!originalWidgetCallback
        });

        nodeType.prototype.onWidgetChanged = function(widget, value) {
            console.log("Widget changed - BEFORE validation:", {
                nodeName: this.title,
                widgetName: widget.name,
                originalValue: value,
                widgetValue: widget.value,
                hasConstraints: !!this.targetConstraints,
                nodeType: this.type,
                constraints: this.targetConstraints
            });

            // For sequence nodes, validate before calling original callback
            if (widget.name === "values" && this.targetConstraints) {
                console.log("About to validate sequence values");
                this.validateSequenceValues(widget);
                // Important: Update the value parameter to match the validated widget value
                value = widget.value;  // Use the validated value
                
                // Force widget update to ensure UI reflects validated value
                widget.value = value;
                if (widget.callback) {
                    widget.callback(value);
                }
                
                console.log("After validation:", {
                    newValue: value,
                    widgetValue: widget.value,
                    originalValue: value
                });
            } else if (["maximum_value", "minimum_value", "starting_value"].includes(widget.name)) {
                console.log("Clamping widget values");
                this.clampWidgetValues();
            }
            
            console.log("Before calling original callback:", {
                finalValue: value,
                widgetValue: widget.value
            });
            
            // Call original callback with validated value
            const result = originalWidgetCallback?.apply(this, [widget, value]);
            
            console.log("After original callback:", {
                result: result,
                finalWidgetValue: widget.value
            });
            
            return result;
        };
    }
}); 