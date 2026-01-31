/* Face Tracker Avatar Demo - MediaPipe + Three.js (local test app) */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  FilesetResolver,
  FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.1.0-alpha-16";

function getViewportSizeAtDepth(camera, depth) {
  const viewportHeightAtDepth =
    2 * depth * Math.tan(THREE.MathUtils.degToRad(0.5 * camera.fov));
  const viewportWidthAtDepth = viewportHeightAtDepth * camera.aspect;
  return new THREE.Vector2(viewportWidthAtDepth, viewportHeightAtDepth);
}

function createCameraPlaneMesh(camera, depth, material) {
  if (camera.near > depth || depth > camera.far) {
    console.warn("Camera plane geometry will be clipped by the `camera`!");
  }
  const viewportSize = getViewportSizeAtDepth(camera, depth);
  const cameraPlaneGeometry = new THREE.PlaneGeometry(
    viewportSize.width,
    viewportSize.height
  );
  cameraPlaneGeometry.translate(0, 0, -depth);
  return new THREE.Mesh(cameraPlaneGeometry, material);
}

class BasicScene {
  constructor() {
    this.height = window.innerHeight;
    this.width = (this.height * 1280) / 720;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      60,
      this.width / this.height,
      0.01,
      5000
    );

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setClearColor(0x000000, 1);
    THREE.ColorManagement.legacy = false;
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    document.body.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x000000);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(0, 1, 0);
    this.scene.add(directionalLight);

    this.camera.position.z = 0;
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    const orbitTarget = this.camera.position.clone();
    orbitTarget.z -= 5;
    this.controls.target = orbitTarget;
    this.controls.update();

    // No video background — floating head on black only

    this.lastTime = performance.now();
    this.callbacks = [];
    this.render();
    window.addEventListener("resize", this.resize.bind(this));
  }

  resize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.render(this.scene, this.camera);
  }

  render(time = this.lastTime) {
    const delta = (time - this.lastTime) / 1000;
    this.lastTime = time;
    for (const callback of this.callbacks) {
      callback(delta);
    }
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame((t) => this.render(t));
  }
}

class Avatar {
  constructor(url, scene) {
    this.url = url;
    this.scene = scene;
    this.loader = new GLTFLoader();
    this.gltf = null;
    this.root = null;
    this.morphTargetMeshes = [];
    this.loadModel(this.url);
  }

  loadModel(url) {
    this.url = url;
    this.loader.load(
      url,
      (gltf) => {
        if (this.gltf) {
          this.gltf.scene.remove();
          this.morphTargetMeshes = [];
        }
        this.gltf = gltf;
        this.scene.add(gltf.scene);
        this.init(gltf);
      },
      (progress) =>
        console.log(
          "Loading model...",
          progress.total ? (100.0 * progress.loaded) / progress.total + "%" : "..."
        ),
      (error) => console.error(error)
    );
  }

  init(gltf) {
    gltf.scene.traverse((object) => {
      if (object.isBone && !this.root) {
        this.root = object;
      }
      if (!object.isMesh) return;
      const mesh = object;
      mesh.frustumCulled = false;
      if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) return;
      this.morphTargetMeshes.push(mesh);
    });
  }

  updateBlendshapes(blendshapes) {
    for (const mesh of this.morphTargetMeshes) {
      if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) continue;
      for (const [name, value] of blendshapes) {
        if (!Object.keys(mesh.morphTargetDictionary).includes(name)) continue;
        const idx = mesh.morphTargetDictionary[name];
        mesh.morphTargetInfluences[idx] = value;
      }
    }
  }

  applyMatrix(matrix, matrixRetargetOptions = {}) {
    const { scale = 1 } = matrixRetargetOptions;
    if (!this.gltf) return;
    matrix.scale(new THREE.Vector3(scale, scale, scale));
    this.gltf.scene.matrixAutoUpdate = false;
    this.gltf.scene.matrix.copy(matrix);
  }

  offsetRoot(offset, rotation) {
    if (this.root) {
      this.root.position.copy(offset);
      if (rotation) {
        const offsetQuat = new THREE.Quaternion().setFromEuler(
          new THREE.Euler(rotation.x, rotation.y, rotation.z)
        );
        this.root.quaternion.copy(offsetQuat);
      }
    }
  }
}

/** 2D image overlay that follows the face (e.g. PNG from your folder). */
class FaceOverlay {
  constructor(scene, imageUrl, options = {}) {
    this.scene = scene;
    this.mesh = null;
    this.scale = options.scale ?? 40;
    const loader = new THREE.TextureLoader();
    loader.load(
      imageUrl,
      (texture) => {
        texture.encoding = THREE.sRGBEncoding;
        const material = new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          side: THREE.DoubleSide,
          depthWrite: false,
        });
        const geometry = new THREE.PlaneGeometry(1, 1);
        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.frustumCulled = false;
        scene.add(this.mesh);
      },
      undefined,
      (err) => console.error("Failed to load face overlay image:", err)
    );
  }

  applyMatrix(matrix, opts = {}) {
    if (!this.mesh) return;
    const scale = opts.scale ?? this.scale;
    const m = matrix.clone().scale(new THREE.Vector3(scale, scale, scale));
    this.mesh.matrixAutoUpdate = false;
    this.mesh.matrix.copy(m);
  }
}

let faceLandmarker = null;
let video = null;
let scene = null;
let avatar = null;
let faceOverlay = null;

function detectFaceLandmarks(time) {
  if (!faceLandmarker || !video) return;
  const landmarks = faceLandmarker.detectForVideo(video, time);

  const transformationMatrices = landmarks.facialTransformationMatrixes;
  if (transformationMatrices && transformationMatrices.length > 0) {
    const matrix = new THREE.Matrix4().fromArray(transformationMatrices[0].data);
    if (faceOverlay) faceOverlay.applyMatrix(matrix, { scale: 40 });
    if (avatar) {
      avatar.applyMatrix(matrix, { scale: 40 });
      const blendshapes = landmarks.faceBlendshapes;
      if (blendshapes && blendshapes.length > 0) {
        avatar.updateBlendshapes(retarget(blendshapes));
      }
    }
  }
}

// Multipliers so mouth/tongue/jaw are more responsive; eyes/brows slightly boosted
const BLENDSHAPE_GAIN = {
  mouth: 2.2,
  jaw: 2.2,
  tongue: 2.5,
  eye: 1.2,
  brow: 1.2,
  default: 1,
};

function retarget(blendshapes) {
  const categories = blendshapes[0].categories;
  const coefsMap = new Map();
  for (let i = 0; i < categories.length; ++i) {
    const blendshape = categories[i];
    const name = blendshape.categoryName;
    let gain = BLENDSHAPE_GAIN.default;
    if (name.includes("mouth") || name.includes("Mouth")) gain = BLENDSHAPE_GAIN.mouth;
    else if (name.includes("jaw") || name.includes("Jaw")) gain = BLENDSHAPE_GAIN.jaw;
    else if (name.includes("tongue") || name.includes("Tongue")) gain = BLENDSHAPE_GAIN.tongue;
    else if (name.includes("eye") || name.includes("Eye")) gain = BLENDSHAPE_GAIN.eye;
    else if (name.includes("brow") || name.includes("Brow")) gain = BLENDSHAPE_GAIN.brow;
    const score = Math.min(1, blendshape.score * gain);
    coefsMap.set(name, score);
  }
  return coefsMap;
}

function onVideoFrame(time) {
  detectFaceLandmarks(time);
  if (video && typeof video.requestVideoFrameCallback === "function") {
    video.requestVideoFrameCallback(onVideoFrame);
  }
}

async function streamWebcamThroughFaceLandmarker() {
  video = document.getElementById("video");

  return new Promise((resolve, reject) => {
    function onAcquiredUserMedia(stream) {
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        video.play().then(() => {
          if (typeof video.requestVideoFrameCallback === "function") {
            video.requestVideoFrameCallback(onVideoFrame);
          } else {
            // Safari / iOS: use requestAnimationFrame instead
            function rafLoop() {
              if (video.readyState >= 2) detectFaceLandmarks(performance.now());
              requestAnimationFrame(rafLoop);
            }
            requestAnimationFrame(rafLoop);
          }
          resolve();
        }).catch(reject);
      };
    }

    navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      })
      .then((evt) => {
        onAcquiredUserMedia(evt);
      })
      .catch((e) => {
        console.error("Failed to acquire camera feed:", e);
        reject(e);
      });
  });
}

function showError(msg) {
  const errEl = document.getElementById("error");
  const infoEl = document.getElementById("info");
  if (errEl) {
    errEl.textContent = msg;
    errEl.className = "error";
  }
  if (infoEl) infoEl.textContent = "";
  console.error(msg);
}

async function runDemo() {
  const info = document.getElementById("info");
  const errorEl = document.getElementById("error");
  if (errorEl) errorEl.textContent = "";

  try {
    info.textContent = "Requesting camera… Allow access when prompted.";
    await streamWebcamThroughFaceLandmarker();

    if (!video) {
      showError("Video element not found.");
      return;
    }

    info.textContent = "Starting 3D view…";
    scene = new BasicScene();
    // 3D avatar for mouth + tongue tracking (blendshapes)
    avatar = new Avatar(
      "https://assets.codepen.io/9177687/raccoon_head.glb",
      scene.scene
    );
    // Optional: 2D overlay (head pose only, no mouth). Set to null to use only the 3D avatar.
    faceOverlay = null;

    info.textContent = "Loading face model… (may take a moment)";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.1.0-alpha-16/wasm"
    );

    const modelPath =
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

    for (const delegate of ["GPU", "CPU"]) {
      try {
        faceLandmarker = await FaceLandmarker.createFromModelPath(vision, modelPath);
        await faceLandmarker.setOptions({
          baseOptions: { delegate },
          runningMode: "VIDEO",
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        });
        break;
      } catch (e) {
        if (delegate === "CPU") throw e;
        console.warn("GPU failed, trying CPU:", e);
      }
    }

    info.textContent = "Ready. Move your face to drive the avatar.";
    console.log("Finished loading MediaPipe model.");
  } catch (e) {
    const msg = e.message || String(e);
    if (msg.includes("Permission") || msg.includes("NotAllowed") || msg.includes("denied")) {
      showError("Camera access denied. Please allow camera and refresh.");
    } else if (msg.includes("NotFound") || msg.includes("DevicesNotFound")) {
      showError("No camera found.");
    } else if (msg.includes("fetch") || msg.includes("network") || msg.includes("Load")) {
      showError("Network error loading model. Check connection and try again.");
    } else {
      showError("Error: " + msg);
    }
  }
}

runDemo();
