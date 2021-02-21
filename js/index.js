const oncapturefaceclick =  async (event) => {
    event.preventDefault();

    // Get Model Type
    var framework = document.getElementById("framework");
    var predictions = null;

    // Get image from canvas
    var canvas = document.getElementById("main-canvas-area");
    var ctx = canvas.getContext("2d");
    
    // Thickness
    try {
        var thickness = document.getElementById('thickness').value;
        var converted_thickness = parseInt(thickness);

        if(converted_thickness < 0 || converted_thickness > canvas.clientWidth || thickness == "") {
            show_alert('Invalid thickness value.');
            return;
        }

        console.log(`Thickness = ${converted_thickness}`);
    } catch(error) {
        show_alert('Thickness value must be number.');
        return;
    }   

    try {
        show_alert('Detecting by ONNX.js. Please wait...');

        // Prepare Image
        console.log('Prepare Image');
        var prepared_data = prepare_image('main-canvas-area');

        predictions = await detection(prepared_data.resize_image, window.session, prepared_data.resize_ratio, onnx_config);
                
        drawoutputtocanvas(predictions, ctx, converted_thickness);   

        alert('Detection complete.');
        show_alert('Detection complete.');
        hide_alert();
	
	var canvas_url = canvas.toDataURL('image/jpg'); 
	var download_image_area = document.getElementById('download-image-area');
	download_image_area.style.display = '';
		
	var download_image_btn = document.getElementById('main-image-area');
	download_image_btn.src = canvas_url;	

	canvas.style.display = 'none';
	document.getElementById('preview-area').style.display = 'none';
    } catch(error) {
        alert('We found a problem during detecting.');
        show_alert('We found a problem during detecting.');
    }
};

const onbrowsechange = (event) => {

    var canvas = document.getElementById('main-canvas-area'),
        context = canvas.getContext('2d');
    var clear_image_area = document.getElementById('clear-image');
    var img = new Image;

    img.onload = function() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        canvas.width = img.width;
        canvas.height = img.height;
        context.drawImage(img, 0, 0);

        canvas.style.display = '';
        clear_image_area.style.display = '';
	    document.getElementById('preview-area').style.display = '';
    }
    
    img.src = URL.createObjectURL(event.target.files[0]);
};

const loadmodel = async () => {
    show_alert('Loading model.');

    try {
        var model_path = onnx_config.model[0];

        // Load model
        console.log(`Model Path = ${model_path}`);
        window.session = new onnx.InferenceSession({ backendHint: 'webgl' });    //wasm
        await window.session.loadModel(model_path); //'./model/ResNet50.onnx');

        show_alert('Finish loading model.');
        alert('Finish Loading Model');
        hide_alert();
    } catch(error) {
        console.error(error);
        alert('We found an error during loading model.');
        show_alert('We found an error during loading model.');
    }
};

const onclearimage = (event) => {
    event.preventDefault();

    var canvas = document.getElementById('main-canvas-area');
    var context = canvas.getContext('2d');
    var browsefile = document.getElementById('browseimage');
    var download_area = document.getElementById('download-image-area');
    var clear_image_area = document.getElementById('clear-image');

    context.clearRect(0, 0, canvas.width, canvas.height);
    
    canvas.style.display = 'none';
    document.getElementById('preview-area').style.display = 'none';
    browsefile.value = '';
    download_area.style.display = 'none';
    clear_image_area.style.display = 'none';
};

const show_alert = (text) => {
    var alert_area = document.getElementById('alert-area');
    var alert_text = document.getElementById('alert-text');

    alert_area.style.display = '';
    alert_text.innerHTML = text;
};

const hide_alert = () => {
    var alert_area = document.getElementById('alert-area');
    alert_area.style.display = 'none';
};

const onLoad = (event) => {
    console.log('Loading Model...');
    loadmodel();
};

var capture_face = document.getElementById('capture-face');
capture_face.addEventListener('click', oncapturefaceclick);

var browsefile = document.getElementById('browseimage');
browsefile.addEventListener('change', onbrowsechange);

window.addEventListener('DOMContentLoaded', onLoad);

var clear_image_area = document.getElementById('clear-image');
clear_image_area.addEventListener('click', onclearimage);

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
    model: ["./model/Mobile0.25.onnx"]
};

//Debugging Purpose
//make_base();
