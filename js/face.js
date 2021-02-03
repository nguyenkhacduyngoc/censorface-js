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

const recover_pad_output = (outputs, pad_params, width, height) => {
    var [img_h, img_w, img_pad_h, img_pad_w] = pad_params;
    var recover_value = tf.tensor([(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]);
    var recover_xy = tf.mul(tf.reshape(outputs.slice([0,0], [-1,14]), [-1, 7, 2]), recover_value).reshape([-1, 14]);

    var bbox = recover_xy.slice([0,0],[-1, 4]);
    var size = bbox.shape[0];
    var landm = recover_xy.slice([0,4],[-1,10]).arraySync();

    var temp_bbox = bbox.arraySync();

    var bbox_result = [];
    var landm_result = [];

    for(var index = 0; index < size; index++) {

        // Bounding Box
        var temp = temp_bbox[index];
        var x1 = temp[0] * width;
        var x2 = temp[1] * height;
        var y1 = temp[2] * width;
        var y2 = temp[3] * height;
        
        bbox_result.push([x1, y1, x2, y2]);

        // Landmark
        landm_result.push([]);

        var landmark = landm[index];
        for(var landm_index = 0; landm_index < landmark.length; landm_index += 2) {
            var x = landmark[landm_index] * width;
            var y = landmark[landm_index + 1] * height;

            landm_result[index].push([x,y]);
        }
    }

    var landm_valid = outputs.slice([0,14],[-1,1]);
    landm_valid = landm_valid.reshape([landm_valid.shape[0]]).arraySync();

    var conf = outputs.slice([0,15],[-1,1]);
    conf = conf.reshape([conf.shape[0]]).arraySync();

    return { 
        bbox: bbox_result, 
        landm: landm_result, 
        landm_valid: landm_valid, 
        conf: conf,
        size: bbox.shape[0]
    };
};

const detect_face = async (canvas_id, scale_down_factor, max_steps, width, height) => {

    // Get the source image
    var src = cv.imread(canvas_id);

    // Scale Down
    var resized_image = resize_image(src, scale_down_factor);

    // Convert to RGB image
    var rgb_image = converttorgb(resized_image);
    resized_image.delete();

    // Pad Image with 800x608?
    var result_image  = resize_image_resolution(rgb_image);
    var result = pad_input_image(result_image, max_steps);
    var { img, param } = result;
    var img_w = img.size().width;
    var img_h = img.size().height;

    // Convert into Tensor with 800x600
    var tensor = convertopencvmattotensor(resize_image_resolution(img));
    result_image.delete();

    //var resize_tensor = tf.image.resizeBilinear(tensor, [800, 600]);
    console.log(`Image is converted into Tensor. Shape = ${tensor.shape[2]},${tensor.shape[1]}`);

    // Read Model and run
    console.log('Loading Model');

    //console.log(respredictionult);
    const resnet_backbone = await tf.loadLayersModel(config.url);
    var detection_result = await detection(tensor, resnet_backbone, config);
    var {bbox, landm, landm_valid, conf, size} = recover_pad_output(detection_result, param, tensor.shape[2], tensor.shape[1]);

    // Debugging Purpose
    /*
    console.log(`Bound Box Result (shape = ${detection_result.bbox.shape})`);
    detection_result.bbox.print();
    console.log(`Landmark Result (shape = ${detection_result.landm.shape})`);
    detection_result.landm.print();
    console.log(`Confidence Result (shape = ${detection_result.conf.shape})`);
    detection_result.conf.print();
    */

    // Dispose
    tensor.dispose();
    detection_result.dispose();

    // Return
    return {bounding_box: bbox, landmark: landm, landmark_valid: landm_valid, conf, size};
};

// Draw in each loop
const drawbbox_landmark = (ctx, output) => {
    var {bounding_box, landmark, landmark_valid, conf, size} = output;
    var context = ctx;

    for(var index = 0; index < size; index++) {
        console.log(`Conf ${index + 1}: ${conf[index]}`);

        // BBox
        var [x1, x2, y1, y2] = bounding_box[index];

        // Draw
        ctx.beginPath();
        ctx.strokeStyle = '#2ecc71';
        ctx.rect(x1, y1, (x2-x1), (y2-y1));
        ctx.stroke();

        // Confidence
        ctx.font = `lighter 12px sans-serif`;
        ctx.textAlign = "left";
        ctx.textBaseline = 'bottom';
        ctx.fillText(`Conf: ${conf[index]}`, x1, y1);

        // Landmark
        if(landmark_valid[index] > 0) 
            for(var landm_index = 0; landm_index < landmark[index].length; landm_index += 2) {
                var [x, y] = landmark[index][landm_index];

                ctx.beginPath();
                ctx.strokeStyle = '#2ecc71';
                ctx.arc(x, y, 1, 0, 2 * Math.PI);
                ctx.stroke();
            }
    }
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
    clip: false,
    weight: "model/weights.json"
};

//"model/weights.json"
//["FPN", "SSH_0", "SSH_1", "SSH_2", "ClassHead_0", "ClassHead_1", "ClassHead_2", "Bbox_0", "Bbox_1", "Bbox_2", "Landmark_0", "Landmark_1", "Landmark_2"]