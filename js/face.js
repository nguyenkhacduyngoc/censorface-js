// Resize the image by resolution 800x600
const resize_image_resolution = (src) => {
    var resized_image = new cv.Mat();
    var dsize = new cv.Size(800, 600);
    cv.resize (src, resized_image, dsize);
    return resized_image;
}

// Resize the image by scale_down_factor
const resize_image = (src, scale_down_factor) => {
    if(scale_down_factor < 1) {
        var resized_image = new cv.Mat();
        var dsize = new cv.Size(0, 0);
        cv.resize (src, resized_image, dsize, scale_down_factor, scale_down_factor, cv.INTER_LINEAR);
        return resized_image;
    }

    return src.clone();;
}

// Convert to RGB image from RGBA (A = alpha).
const converttorgb = (src) => {
    var rgb_image = new cv.Mat();
    cv.cvtColor(src, rgb_image, cv.COLOR_RGBA2RGB, 0); 
    return rgb_image;
};

const channelmean = (img) => {
    var cols = img.cols;
    var rows = img.rows;
    var size = cols * rows;
    var channels = img.channels();
    var mean = [0.0, 0.0, 0.0];   //R,G,B

    for(var r = 0; r < rows; r++)
        for(var c = 0; c < cols; c++) 
            for(var i = 0; i < channels; i++) 
                mean[i] += img.data[r * cols * channels + c * channels + i];

    //mean[1] += img.data[r * cols * channels + c * channels + 1];
    //mean[2] += img.data[r * cols * channels + c * channels + 2];

    for(var c = 0; c < channels; c++)
        mean[c] /= size;

    return mean;
};

const convertopencvmattotensor = (img) => {
    var cols = img.cols;
    var rows = img.rows;
    var channels = img.channels();

    var img_data = [];

    for(var r = 0; r < rows; r++) {
        img_data.push([]);
        for(var c = 0; c < cols; c++) {
            img_data[r].push([]);
            for(var i = 0; i < channels; i++)
                img_data[r][c].push(img.data[r * cols * channels + c * channels + i])
        }
    }
    var img_tensor = tf.tensor4d([img_data], [1, rows, cols, 3]);

    return img_tensor;
};

// Pad input images with specified steps
const pad_input_image = (img, max_steps) => {
    // Pad Image to suitable shape
    var img_w = img.size().width;
    var img_h = img.size().height;

    var img_pad_h = 0;
    if(img_h % max_steps > 0)
        img_pad_h = max_steps - img_h % max_steps;

    var img_pad_w = 0;
    if(img_w % max_steps > 0)
        img_pad_w = max_steps - img_w % max_steps;

    var result_img = new cv.Mat();
    var padd_value = channelmean(img); //BGR

    //padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    cv.copyMakeBorder(img, result_img, 0, img_pad_h, 0, img_pad_w,
        cv.BORDER_CONSTANT, new cv.Scalar(padd_value[0], padd_value[1], padd_value[2]));

    var pad_params = [img_h, img_w, img_pad_h, img_pad_w];

    return {img: result_img, param: pad_params, mean: padd_value}
};

const load_model = (MODEL_URL, filename) => 
    new Promise(async (resolve) => {
        let file = await fetch(MODEL_URL).then(r => r.blob());
        let path = filename;
        let reader = new FileReader();
        
        reader.readAsArrayBuffer(file);
        reader.onload = function(ev) {
            if (reader.readyState === 2) {
                let buffer = reader.result;
                let data = new Uint8Array(buffer);
                cv.FS_createDataFile('/', path, data, true, false, false);
                resolve(path);
            }
        }
    });

const recover_pad_output = (outputs, pad_params) => {
    var [img_h, img_w, img_pad_h, img_pad_w] = pad_params;
    var recover_value = tf.tensor([(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]);
    var recover_xy = tf.mul(tf.reshape(outputs.slice([0,0], [-1,14]), [-1, 7, 2]), recover_value).reshape([-1, 14]);

    var bbox = recover_xy.slice([0,0],[-1, 4])
    var landm = recover_xy.slice([0,4],[-1,10]);

    var landm_valid = outputs.slice([0,14],[-1,1]);
    var conf = outputs.slice([0,15],[-1,1]);

    return { bbox, landm, landm_valid, conf };
};

const detect_face = async (canvas_id, scale_down_factor, max_steps) => {

    // Get the source image
    var src = cv.imread(canvas_id);

    // Scale Down
    var resized_image = resize_image(src, scale_down_factor);

    // Convert to RGB image
    var rgb_image = converttorgb(resized_image);
    resized_image.delete();

    // Pad Image
    var result = pad_input_image(rgb_image, max_steps);
    var { img, param } = result;

    // Convert into Tensor
    result_image  = resize_image_resolution(img);
    var tensor = convertopencvmattotensor(result_image);
    result_image.delete();

    //var resize_tensor = tf.image.resizeBilinear(tensor, [800, 600]);
    console.log(`Image is converted into Tensor. Shape = ${tensor.shape[2]},${tensor.shape[1]}`);

    // Read Model and run
    const resnet_backbone = await tf.loadLayersModel(config.url);
    var detection_result = await detection(tensor, resnet_backbone, config);
    var output = recover_pad_output(detection_result, param);
    console.log(`Padding parameter = ${param}`);
    // Debugging Purpose
    /*
    console.log(`Bound Box Result (shape = ${detection_result.bbox.shape})`);
    detection_result.bbox.print();
    console.log(`Landmark Result (shape = ${detection_result.landm.shape})`);
    detection_result.landm.print();
    console.log(`Confidence Result (shape = ${detection_result.conf.shape})`);
    detection_result.conf.print();
    */

    var bounding_box = output.bbox.array();
    var landmark = output.landm.array();
    var landmark_valid = output.landm_valid.array();
    var conf = output.conf.array();

    // Dispose
    tensor.dispose();

    // Return
    return {bounding_box, landmark, landmark_valid, conf, size: output.bbox.shape[0]}
};

// Configuration
const config = {
    input_size: [800, 600],
    min_size: [[16, 32], [64, 128], [256, 512]],
    out_ch: 256,
    url: 'model/Backbone/model.json',
    steps: [8, 16, 32],
    variances: [0.1, 0.2],
    iou_thresh: 0.4,
    score_thresh: 0.02,
    clip: false
};