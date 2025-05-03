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
            let mediaRecorder = null;
            let recordedChunks = [];
            
            // Initialize webcam once for all nodes
            const initWebcam = async () => {
                // console.log("Initializing global webcam stream...");
                if (!webcamStream) {
                    try {
                        webcamStream = await navigator.mediaDevices.getUserMedia({
                            video: true,
                            audio: true  // Now requesting audio too for video recording
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

            // Function to record video for a specified duration
            const recordVideo = async (duration) => {
                return new Promise(async (resolve, reject) => {
                    try {
                        // Initialize webcam if not already done
                        const { stream, video } = await initWebcam();
                        
                        // Clear previous recording data
                        recordedChunks = [];
                        
                        // Initialize MediaRecorder
                        mediaRecorder = new MediaRecorder(stream, {
                            mimeType: 'video/webm;codecs=vp9,opus'
                        });
                        
                        // Handle data available event
                        mediaRecorder.ondataavailable = (event) => {
                            if (event.data.size > 0) {
                                recordedChunks.push(event.data);
                            }
                        };
                        
                        // Handle recording stop event
                        mediaRecorder.onstop = () => {
                            // Create a blob from recorded chunks
                            const blob = new Blob(recordedChunks, { type: 'video/webm' });
                            
                            // Convert blob to base64 data URL
                            const reader = new FileReader();
                            reader.onloadend = () => {
                                const base64data = reader.result;
                                resolve(base64data);
                            };
                            reader.readAsDataURL(blob);
                        };
                        
                        // Start recording
                        mediaRecorder.start();
                        console.log(`Recording started for ${duration} seconds`);
                        
                        // Stop recording after specified duration
                        setTimeout(() => {
                            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                                mediaRecorder.stop();
                                console.log('Recording stopped');
                            }
                        }, duration * 1000);
                    } catch (error) {
                        console.error("Error during video recording:", error);
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
                        console.log("Captured frame at native resolution:", canvas.width, "x", canvas.height);
                        return dataUrl;
                    } 
                    // Video recording mode
                    else {
                        console.log(`Starting video recording for ${recordSeconds} seconds`);
                        
                        // Record video and return data URL
                        const videoDataUrl = await recordVideo(recordSeconds);
                        console.log("Video recording completed, sending data to backend");
                        return videoDataUrl;
                    }
                };

                return r;
            };
        }
    }
}); 