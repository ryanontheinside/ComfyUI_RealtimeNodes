// Widget validation for RealTime nodes
import { app } from "../../../scripts/app.js";

// TODO: Add validation for step values to ensure they match target widget constraints


// Register validation behavior when nodes are connected
app.registerExtension({
    name: "RealTime.WidgetValidation",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData.category?.startsWith("real-time/control/")) {
            return;
        }

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
                    if (this.targetConstraints) {
                        this.validateSequenceValues(valuesWidget);
                        value = valuesWidget.value;
                    }
                    return originalCallback?.call(valuesWidget, value);
                };
            }
            
            // Function to check connections
            const checkConnections = () => {
                const outputLinks = this.outputs[0]?.links || [];
                
                for (const linkId of outputLinks) {
                    const link = app.graph.links[linkId];
                    if (!link) continue;

                    const targetNode = app.graph.getNodeById(link.target_id);
                    const targetSlot = link.target_slot;
                    
                    if (targetNode?.widgets) {
                        const inputName = targetNode.inputs[targetSlot]?.name;
                        const targetWidget = targetNode.widgets.find(w => w.name === inputName);
                        
                        if (targetWidget?.options || (targetWidget?.type === "converted-widget" && targetWidget.options)) {
                            this.targetConstraints = {
                                min: targetWidget.options.min,
                                max: targetWidget.options.max,
                                step: targetWidget.options.step
                            };

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

            // Check connections at different intervals to ensure graph is loaded
            checkConnections();
            setTimeout(checkConnections, 100);
            setTimeout(checkConnections, 1000);
            
            return result;
        };

        const originalOnConnectOutput = nodeType.prototype.onConnectOutput;
        nodeType.prototype.onConnectOutput = function (slot, type, input, targetNode, targetSlot) {
            const result = originalOnConnectOutput?.apply(this, arguments);

            if (targetNode?.widgets) {
                const inputName = targetNode.inputs[targetSlot]?.name;
                const targetWidget = targetNode.widgets.find(w => w.name === inputName);
                
                if (targetWidget?.options || (targetWidget?.type === "converted-widget" && targetWidget.options)) {
                    const options = targetWidget.options;
                    this.targetConstraints = {
                        min: options.min,
                        max: options.max,
                        step: options.step
                    };

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

            this.clampWidgetValues();
        };

        // Add method to clamp widget values
        nodeType.prototype.clampWidgetValues = function() {
            const minWidget = this.widgets.find(w => w.name === "minimum_value");
            const maxWidget = this.widgets.find(w => w.name === "maximum_value");
            const startWidget = this.widgets.find(w => w.name === "starting_value");

            if (minWidget && maxWidget) {
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
            if (!this.targetConstraints) return;

            const separator = widget.options?.multiline ? "\n" : ",";
            const values = widget.value.split(separator)
                .map(v => v.trim())
                .filter(v => v.length > 0);

            const isFloat = this.type === "FloatSequence";
            const isInt = this.type === "IntSequence";

            if (isFloat || isInt) {
                const validValues = values.map(v => {
                    let num = isFloat ? parseFloat(v) : parseInt(v);
                    if (isNaN(num)) num = this.targetConstraints.min;
                    const clamped = Math.max(this.targetConstraints.min, Math.min(num, this.targetConstraints.max));
                    return isFloat ? clamped.toString() : Math.round(clamped).toString();
                });

                widget.value = validValues.join(separator);
            }
        };

        // Override the widget's callback to enforce constraints
        const originalWidgetCallback = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function(widget, value) {
            if (widget.name === "values" && this.targetConstraints) {
                this.validateSequenceValues(widget);
                value = widget.value;
                
                widget.value = value;
                if (widget.callback) {
                    widget.callback(value);
                }
            } else if (["maximum_value", "minimum_value", "starting_value"].includes(widget.name)) {
                this.clampWidgetValues();
            }
            
            return originalWidgetCallback?.apply(this, [widget, value]);
        };
    }
}); 