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

const recover_pad_output = (outputs, pad_params, width, height) => { //, width, height) => {
    var [img_h, img_w, img_pad_h, img_pad_w] = pad_params;
    var recover_value = tf.tensor([(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]);
    var recover_xy = tf.mul(tf.reshape(outputs.slice([0,0], [-1,14]), [-1, 7, 2]), recover_value).reshape([-1, 14]);

    var bbox = recover_xy.slice([0,0],[-1, 4]);
    var size = outputs.shape[0];
    //var landm = recover_xy.slice([0,4],[-1,10]).arraySync();

    var bbox_result = [];
    var landm_result = [];

    for(var index = 0; index < size; index++) {

        // Bounding Box
        var temp = recover_xy.slice([index,0],[-1, 4]).dataSync();

        var x1 = temp[0] * width;
        var x2 = temp[1] * height;
        var y1 = temp[2] * width;
        var y2 = temp[3] * height;
        
        bbox_result.push([x1, y1, x2, y2]);

        // Landmark
        landm_result.push([]);

        var landmark = recover_xy.slice([index,4],[-1, 10]).dataSync();
        for(var landm_index = 0; landm_index < landmark.length; landm_index += 2) {
            var x = landmark[landm_index] * width;
            var y = landmark[landm_index + 1] * height;

            landm_result[index].push([x,y]);
        }
    }

    var landm_valid = outputs.slice([0,14],[-1,1]);
    landm_valid = landm_valid.reshape([landm_valid.shape[0]]).dataSync();

    var conf = outputs.slice([0,15],[-1,1]);
    conf = conf.reshape([conf.shape[0]]).dataSync();

    var output = { 
        bbox: bbox_result, 
        landm: landm_result, 
        landm_valid: landm_valid, 
        conf: conf,
        size: bbox.shape[0]
    };

    return output;
};

// TF NMS
const tf_nms = (output, iou_thresh) => {
    var { max, min } = Math;
    var {bbox, landm, landm_valid, conf, size} = output;
    var foundLocations = [];
    var pick = [];

    for(var i = 0; i < size; i++) {
        var x1 = bbox[i][0],
            y1 = bbox[i][1],
            x2 = bbox[i][2],
            y2 = bbox[i][3];

        var width = x2 - x1,
            height = y2 - y1;
        
        if(width > 0 && height > 0) {
            var area = width * height;
            foundLocations.push({x1,y1,x2,y2,width,height,area,index: i});
        }
    }

    foundLocations.sort((b1, b2) => {
        return b1.y2 - b2.y2;
    });

    while(foundLocations.length > 0) {
        var last = foundLocations[foundLocations.length - 1];
        var suppress = [last];
        pick.push(foundLocations.length - 1);

        for(let i = 0; i < foundLocations.length; i++) {
            const box = foundLocations[i];
            const xx1 = max(box.x1, last.x1);
            const yy1 = max(box.y1, last.y1);
            const xx2 = min(box.x2, last.x2);
            const yy2 = min(box.y2, last.y2);
            const w = max(0, xx2 - xx1);
            const h = max(0, yy2 - yy1);
            const overlap = (w*h) / box.area;

            if(overlap > iou_thresh)
                suppress.push(foundLocations[i]);
        }

        foundLocations = foundLocations.filter((box) => {
            return !suppress.find((supp) => {
                return supp === box;
            })
        });
    }

    var result_bbox = [],
        result_scores = [],
        result_landms = [],
        result_valid = [];

    pick.forEach((pick_index, i) => {
        result_bbox.push([]);
        for(var j = 0; j < 4; j++)
            result_bbox[i].push(bbox[pick_index][j]);
        
        result_scores.push(conf[pick_index]);
        result_valid.push(landm_valid[pick_index]);

        result_landms.push([]);
        for(var j = 0; j < 10; j++)
            result_landms[i].push(landm[pick_index][j]);
    });

    return {bbox: result_bbox, landm: result_landms, landm_valid: result_valid, conf: result_scores, size: pick.length};
};

const detect_face = async (canvas_id, resnet_backbone, scale_down_factor, max_steps, width, height, config) => {

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
    var tensor = convertopencvmattotensor(img);
    result_image.delete();

    //var resize_tensor = tf.image.resizeBilinear(tensor, [800, 600]);
    console.log(`Image is converted into Tensor. Shape = ${tensor.shape[2]},${tensor.shape[1]}`);

    // Read Model and run
    console.log('Loading Model');

    //console.log(respredictionult);
    var detection_result = await tensorflow_detection(tensor, resnet_backbone, config);

    var recovered_output = recover_pad_output(detection_result, param, width, height); //, tensor.shape[2], tensor.shape[1]);
    var {bbox, landm, landm_valid, conf, size} = tf_nms(recovered_output, config.iou_thresh);

    // Dispose
    tensor.dispose();
    detection_result.dispose();

    // Return
    return {bounding_box: bbox, landmark: landm, landmark_valid: landm_valid, conf, size};
};

// Draw in each loop
const drawbbox_landmark = (ctx, output) => {
    var {bounding_box, landmark, landmark_valid, conf, size} = output;

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