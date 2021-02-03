const onmainbuttonclick = async (event) => {

    // Main video area
    var videoinput = document.getElementById('main-video-area');

    // Setting page size
    videoinput.width = 800;
    videoinput.height = 600;

    // Open web camera
    var stream = await navigator.mediaDevices.getUserMedia({ video: { width: 800, height: 600 }, audio: true });
    videoinput.srcObject = stream;

    // Play video from webcam
    videoinput.play();
};

const drawcanvas_videoframe = () => {
    var videoinput = document.getElementById('main-video-area');
    var canvas = document.getElementById('main-canvas-area');
    var ctx = canvas.getContext('2d');

    if(videoinput.paused || videoinput.ended) return false;

    canvas.width = videoinput.width + 10;
    canvas.height = videoinput.height + 10;
    canvas.style.width = videoinput.width + 10;
    canvas.style.height = videoinput.height + 10;

    // Draw Video
    ctx.drawImage(videoinput, 0, 0, canvas.width, canvas.height);
    
    requestAnimationFrame(drawcanvas_videoframe);
};

const onvideotagplaying = (event) => {
    requestAnimationFrame(drawcanvas_videoframe);
};

const onstopbuttonclick = (event) => {
    var videoinput = document.getElementById('main-video-area');
    var stop_button = document.getElementById('stop-video');
 
    if (!videoinput.paused) {
        videoinput.pause();
        stop_button.innerText = 'Play Camera';
    } else {
        videoinput.play();
        stop_button.innerText = 'Pause Camera';
    }
};

const oncapturefaceclick =  async (event) => {
    event.preventDefault();

    // Get Mode lType
    var framework = document.getElementById("framework");
    var predictions = null;

    // Get image from canvas
    var canvas = document.getElementById("main-canvas-area");
    var ctx = canvas.getContext("2d");

    if(framework.value == '0' ) {        // ONNX
        // Prepare Image
        console.log('Prepare Image');
        var prepared_data = prepare_image('main-canvas-area');

        predictions = await detection(prepared_data.resize_image, window.session, prepared_data.resize_ratio, onnx_config);
            
        drawoutputtocanvas(predictions, ctx);    
    } else if(framework.value == '1') {     // Tensorflow.js
        predictions = await detect_face('main-canvas-area', window.resnet_backbone, 1, 32, canvas.width, canvas.height, tensorflow_config);

        drawbbox_landmark(ctx, predictions);
    }
};

const onbrowsechange = (event) => {

    var canvas = document.getElementById('main-canvas-area'),
        context = canvas.getContext('2d');
    var img = new Image;
    img.onload = function() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        canvas.width = img.width;
        canvas.height = img.height;
        context.drawImage(img, 0, 0);
    }
    
    img.src = URL.createObjectURL(event.target.files[0]);
};

const onloadmodelclick = async (event) => {
    if(framework.value == '0') {        // ONNX
        // Model Selection
        var model_select = document.getElementById('model');
        var model_path = onnx_config.model[parseInt(model_select.value)];

        // Load model
        console.log(`Model Path = ${model_path}`);
        window.session = new onnx.InferenceSession({ backendHint: 'wasm' });    //wasm
        await window.session.loadModel(model_path); //'./model/ResNet50.onnx');
    } else if(framework.value == '1') {     // Tensorflow.js
        window.resnet_backbone = await tf.loadLayersModel(tensorflow_config.url);
    }

    console.log('Finish Loading Model');
};

async function make_base()
{
    var canvas = document.getElementById('main-canvas-area'),
    context = canvas.getContext('2d');
    base_image = new Image(800, 450);
    canvas.width = 800;
    canvas.height = 450;
    base_image.src = 'pic/prayuth.jpg';
    base_image.onload = function()  {
        context.drawImage(base_image, 0, 0);
    }
}

var capture_face = document.getElementById('capture-face');
capture_face.addEventListener('click', oncapturefaceclick);

var browsefile = document.getElementById('browseimage');
browsefile.addEventListener('change', onbrowsechange);

var load_model_btn = document.getElementById('load-model');
load_model_btn.addEventListener('click', onloadmodelclick);

var onnx_config = {
    name: 'Resnet50',
    min_sizes: [[16, 32], [64, 128], [256, 512]],
    steps: [8, 16, 32],
    variance: [0.1, 0.2],
    clip: false,
    loc_weight: 2.0,
    gpu_train: true,
    batch_size: 24,
    ngpu: 4,
    epoch: 100,
    decay1: 70,
    decay2: 90,
    image_size: 800,
    pretrain: true,
    return_layers: {'layer2': 1, 'layer3': 2, 'layer4': 3},
    in_channel: 256,
    out_channel: 256,
    confidence_threshold: 0.02,
    top_k: 5000,
    keep_top_k: 740,
    nms_threshold: 0.4,
    model: ["./model/ResNet50.onnx", "./model/Mobile0.25.onnx"]
};

// Configuration
var tensorflow_config = {
    input_size: [800, 600],
    min_size: [[16, 32], [64, 128], [256, 512]],
    out_ch: 256,
    url: 'model/backbone_model/model.json',
    steps: [8, 16, 32],
    variances: [0.1, 0.2],
    iou_thresh: 0.4,
    score_thresh: 0.02,
    top_k: 5000,
    clip: false,
    weight: "model/weights_total.json"
};


make_base();