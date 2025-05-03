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
                            audio: false  // No need for audio anymore
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

            // Function to capture frames for a given duration
            const captureFrames = async (duration, fps = 30) => {
                return new Promise(async (resolve, reject) => {
                    try {
                        // Initialize webcam if not already done
                        const { video } = await initWebcam();
                        
                        // Create canvas for frame capture
                        const canvas = document.createElement('canvas');
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        const ctx = canvas.getContext('2d');
                        
                        // Calculate total frames to capture
                        const totalFrames = Math.ceil(duration * fps);
                        const frameInterval = 1000 / fps; // ms between frames
                        
                        console.log(`Capturing ${totalFrames} frames at ${fps} FPS (${frameInterval}ms intervals)`);
                        
                        // Store captured frames
                        const frames = [];
                        
                        // Start capturing frames at regular intervals
                        let frameCount = 0;
                        const startTime = Date.now();
                        
                        const captureFrame = () => {
                            // Draw current video frame to canvas
                            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                            const frameDataUrl = canvas.toDataURL('image/jpeg', 0.95);
                            frames.push(frameDataUrl);
                            
                            frameCount++;
                            if (frameCount < totalFrames) {
                                // Calculate next frame time based on start time to avoid drift
                                const nextFrameTime = startTime + Math.round(frameCount * frameInterval);
                                const delay = Math.max(0, nextFrameTime - Date.now());
                                
                                // Schedule next frame capture
                                setTimeout(captureFrame, delay);
                            } else {
                                // All frames captured
                                console.log(`Completed capture of ${frames.length} frames`);
                                
                                resolve({
                                    frames: frames,
                                    width: canvas.width,
                                    height: canvas.height,
                                    duration: duration,
                                    fps: fps
                                });
                            }
                        };
                        
                        // Start capture process
                        captureFrame();
                    } catch (error) {
                        console.error("Error during capture:", error);
                        reject(error);
                    }
                });
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
                const canvas = document.createElement("canvas");
                
                camera.serializeValue = async () => {
                    // Only initialize webcam when actually needed
                    if (!webcamVideo || webcamVideo.paused || webcamVideo.readyState !== 4) {
                        // console.log("Initializing webcam for first use...");
                        const { video } = await initWebcam();
                        webcamVideo = video;
                    }
                    
                    // Find record_seconds input value
                    const recordSecondsWidget = this.widgets.find(w => w.name === "record_seconds");
                    const recordSeconds = recordSecondsWidget ? recordSecondsWidget.value : 0;
                    
                    // Single image capture mode
                    if (recordSeconds <= 0) {
                        console.log("Capturing single frame from video:", {
                            videoWidth: webcamVideo.videoWidth,
                            videoHeight: webcamVideo.videoHeight
                        });
                        
                        // Always use native webcam resolution
                        canvas.width = webcamVideo.videoWidth;
                        canvas.height = webcamVideo.videoHeight;
                        
                        const ctx = canvas.getContext("2d");
                        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
                        
                        const dataUrl = canvas.toDataURL("image/png");
                        return dataUrl;
                    } 
                    // Video capture mode - capture frames
                    else {
                        
                        // Capture frames
                        const captureResult = await captureFrames(recordSeconds);
                        
                        // Convert to JSON and return
                        return JSON.stringify(captureResult);
                    }
                };

                return r;
            };
        }
    }
}); 