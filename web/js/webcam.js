import { app } from "../../../scripts/app.js";
//TODO: fix  occlusion of dimension editing popups
app.registerExtension({
    name: "Core.FastWebcam",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FastWebcamCapture") {
            console.log("Registering FastWebcamCapture node type");
            
            // Store webcam stream at extension level to avoid reinitializing
            let webcamStream = null;
            let webcamVideo = null;
            
            // Initialize webcam once for all nodes
            const initWebcam = async () => {
                // console.log("Initializing global webcam stream...");
                if (!webcamStream) {
                    try {
                        webcamStream = await navigator.mediaDevices.getUserMedia({
                            video: true,
                            audio: false
                        });
                        // console.log("Got media stream:", webcamStream);
                        
                        webcamVideo = document.createElement("video");
                        webcamVideo.srcObject = webcamStream;
                        webcamVideo.style.display = 'none';
                        document.body.appendChild(webcamVideo);
                        await webcamVideo.play();
                        
                        console.log("Video playing. Video state:", {
                            paused: webcamVideo.paused,
                            readyState: webcamVideo.readyState,
                            videoWidth: webcamVideo.videoWidth,
                            videoHeight: webcamVideo.videoHeight
                        });
                    } catch (error) {
                        // console.error("Webcam initialization error:", error);
                        throw error;
                    }
                }
                return { stream: webcamStream, video: webcamVideo };
            };
            
            nodeType.prototype.getCustomWidgets = function() {
                // console.log("Getting custom widgets");
                return {
                    WEBCAM: (node, inputName) => {
                        // console.log("Creating WEBCAM widget");
                        const container = document.createElement("div");
                        container.style.background = "rgba(0,0,0,0.25)";
                        container.style.textAlign = "center";
                        
                        const statusLabel = document.createElement("div");
                        statusLabel.textContent = "Webcam Ready";
                        container.appendChild(statusLabel);
                        
                        const widget = node.addDOMWidget(inputName, "WEBCAM", container);
                        return { widget };
                    }
                };
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                // console.log("Node created");
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const camera = this.widgets.find((w) => w.name === "image");
                const w = this.widgets.find((w) => w.name === "width");
                const h = this.widgets.find((w) => w.name === "height");
                
                const canvas = document.createElement("canvas");
                
                camera.serializeValue = async () => {
                    // console.log("Serializing camera value");
                    
                    // Only initialize webcam when actually needed
                    if (!webcamVideo || webcamVideo.paused || webcamVideo.readyState !== 4) {
                        // console.log("Initializing webcam for first use...");
                        const { video } = await initWebcam();
                        webcamVideo = video;
                    }
                    
                    console.log("Capturing frame from video:", {
                        paused: webcamVideo.paused,
                        readyState: webcamVideo.readyState,
                        videoWidth: webcamVideo.videoWidth,
                        videoHeight: webcamVideo.videoHeight
                    });
                    
                    canvas.width = w.value || webcamVideo.videoWidth || 640;
                    canvas.height = h.value || webcamVideo.videoHeight || 480;
                    
                    const ctx = canvas.getContext("2d");
                    ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
                    
                    const dataUrl = canvas.toDataURL("image/png");
                    // console.log("Successfully captured and converted frame to data URL");
                    return dataUrl;
                };

                return r;
            };
        }
    }
}); 