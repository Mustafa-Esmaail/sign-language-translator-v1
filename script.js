// Copyright 2022 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.1.0-alpha-11";
const demosSection = document.getElementById("demos");
const classesNames = [
  "ع",
  "ال",
  "ا",
  "ب",
  "د",
  "ظ",
  "ض",
  "ف",
  "ق",
  "غ",
  "ه",
  "ح",
  "ج",
  "ك",
  "خ",
  "لا",
  "ل",
  "م",
  "nothing",
  "ن",
  "ر",
  "ص",
  "س",
  "ش",
  "ت",
  "ط",
  "ث",
  "ذ",
  "ة",
  "و",
  "ي",
  "ئ",
  "ز",
];
let spans = document.getElementsByClassName("img-result");
let cam = document.getElementsByClassName("cam-result");
let word = "";
let handLandmarker = undefined;
let handModel = undefined;
let Camletter = [];
let Imgletter = [];
let runningMode = "IMAGE";
let enableWebcamButton;
let Space=false;
let webcamRunning = false;

const videoHeight = 360;
const videoWidth = 480;
// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createHandLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.1.0-alpha-11/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task`,
    },
    runningMode: runningMode,
    numHands: 2,
  });
  handModel = await tf.loadLayersModel(
    "https://raw.githubusercontent.com/Mustafa-Esmaail/arabic-sign-language/sign-lang-model-v1/model.json"
  );
  demosSection.classList.remove("invisible");
};
createHandLandmarker();

/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/
// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.
const imageContainers = document.getElementsByClassName("detectOnClick");
// Now let's go through all of these and add a click event listener.
for (let i = 0; i < imageContainers.length; i++) {
  // Add event listener to the child element whichis the img element.
  imageContainers[i].children[0].addEventListener("click", handleClick);
}
// When an image is clicked, let's detect it and display results!
async function handleClick(event) {
  if (!handLandmarker) {
    console.log("Wait for handLandmarker to load before clicking!");
    return;
  }
  if (runningMode === "VIDEO") {
    runningMode = "IMAGE";
    await handLandmarker.setOptions({ runningMode: "IMAGE" });
  }
  // Remove all landmarks drawed before
  const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
  for (var i = allCanvas.length - 1; i >= 0; i--) {
    const n = allCanvas[i];
    n.parentNode.removeChild(n);
  }
  // We can call handLandmarker.detect as many times as we like with
  // different image data each time. This returns a promise
  // which we wait to complete and then call a function to
  // print out the results of the prediction.
  const handLandmarkerResult = handLandmarker.detect(event.target);
  console.log(event.target.naturalHeight);

  handLandmarkerResult.landmarks.map((landmarks) => {
    console.log(landmarks);
    // let landmark_list = calc_landmark_list(
    //     videoWidth,
    //     videoHeight,
    //   landmarks
    // );
    let landmark_point = [];

    landmarks.map((landmark) => {
      const landmark_x = Math.min(
        Number(landmark.x * event.target.naturalWidth),
        event.target.naturalWidth - 1
      );
      const landmark_y = Math.min(
        Number(landmark.y * event.target.naturalHeight),
        event.target.naturalHeight - 1
      );
      landmark_point.push([landmark_x, landmark_y]);
    });

    var base_x = 0;
    var base_y = 0;
    let marks = [];
    console.log(landmark_point);

    landmark_point.map((point, index) => {
      if (index === 0) {
        base_x = landmark_point[index][0];

        base_y = landmark_point[index][1];
      }
      landmark_point[index][0] = landmark_point[index][0] - base_x;
      landmark_point[index][1] = landmark_point[index][1] - base_y;
      marks.push(landmark_point[index][0]);
      marks.push(landmark_point[index][1]);
    });

    let max_value = Math.max.apply(null, marks.map(Math.abs));

    marks.map((point, idx) => {
      marks[idx] = marks[idx] / max_value;
    });

    console.log(marks);

    let tfMark = tf.tensor(marks).reshape([1, 42]);

    const prediction = handModel.predict(tfMark);
    const handResult = prediction.dataSync();
    const arr = Array.from(handResult);
    const maxPredict = Math.max.apply(null, arr);
    const idx = arr.indexOf(maxPredict);
    // console.log(handResult);
    // console.log(arr);
    // console.log(maxPredict);
    // console.log(idx);

    Imgletter.push(classesNames[idx]);
    // console.log(Imgletter);
    // console.log(event.target.id);
    // console.log()
    spans[event.target.id - 1].innerHTML = classesNames[idx];
  });
  const canvas = document.createElement("canvas");
  canvas.setAttribute("class", "canvas");
  canvas.setAttribute("width", event.target.naturalWidth + "px");
  canvas.setAttribute("height", event.target.naturalHeight + "px");
  canvas.style =
    "left: 0px;" +
    "top: 0px;" +
    "width: " +
    event.target.width +
    "px;" +
    "height: " +
    event.target.height +
    "px;";
  event.target.parentNode.appendChild(canvas);
  const cxt = canvas.getContext("2d");
  for (const landmarks of handLandmarkerResult.landmarks) {
    drawConnectors(cxt, landmarks, HAND_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 5,
    });
    drawLandmarks(cxt, landmarks, { color: "#FF0000", lineWidth: 1 });
  }
}
/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
// Check if webcam access is supported.
const hasGetUserMedia = () => {
  var _a;
  return !!((_a = navigator.mediaDevices) === null || _a === void 0
    ? void 0
    : _a.getUserMedia);
};
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!handLandmarker) {
    console.log("Wait! objectDetector not loaded yet.");
    return;
  }
  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    // let addSpace = document.getElementById("addSpace");
    // cam.innerHTML=Camletter.join('')

    
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICITONS";
  }
  // getUsermedia parameters.
  const constraints = {
    video: true,
  };
  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}
async function predictWebcam() {
  canvasElement.style.height = `${videoHeight}px`;
  video.style.height = `${videoHeight}px`;
  canvasElement.style.width = `${videoWidth}px`;
  video.style.width = `${videoWidth}px`;
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();
  const results = handLandmarker.detectForVideo(video, startTimeMs);

  results.landmarks.map((landmarks) => {
    let landmark_point = [];

    landmarks.map((landmark) => {

      const landmark_x = Math.min(
        Number(landmark.x * videoWidth),
        videoWidth - 1
      );
      const landmark_y = Math.min(
        Number(landmark.y * videoHeight),
        videoHeight - 1
      );
      landmark_point.push([landmark_x, landmark_y]);
    });

    var base_x = 0;
    var base_y = 0;
    let marks = [];

    landmark_point.map((point, index) => {
      if (index === 0) {
        base_x = landmark_point[index][0];

        base_y = landmark_point[index][1];
      }
      landmark_point[index][0] = landmark_point[index][0] - base_x;
      landmark_point[index][1] = landmark_point[index][1] - base_y;
      marks.push(landmark_point[index][0]);
      marks.push(landmark_point[index][1]);
    });

    let max_value = Math.max.apply(null, marks.map(Math.abs));

    marks.map((point, idx) => {
      marks[idx] = marks[idx] / max_value;
    });

    let tfMark = tf.tensor(marks).reshape([1, 42]);

    const prediction = handModel.predict(tfMark);
    const handResult = prediction.dataSync();
    const arr = Array.from(handResult);
    const maxPredict = Math.max.apply(null, arr);
    const idx = arr.indexOf(maxPredict);
    Camletter.push(classesNames[idx]);

  });

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (results.landmarks) {
    for (const landmarks of results.landmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 5,
      });
      drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });
    }
  }

  canvasCtx.restore();
  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    word = Camletter.join("");
    console.log(Camletter)
    let addSpace = document.getElementById("addSpace");
    addSpace.addEventListener('click',()=>{
      Camletter.push(' ')
    })
    let delChar = document.getElementById("delChar");
    delChar.addEventListener('click',()=>{
      Camletter[Camletter.length-1]='';
    })
    let delAll = document.getElementById("delAll");
    delAll.addEventListener('click',()=>{
      Camletter.splice(0, Camletter.length, );
    })
    console.log(Camletter)

    document.querySelector(".cam-result").innerHTML=word
    setTimeout(() => {
      window.requestAnimationFrame(predictWebcam);
    }, 500);
  }
}
