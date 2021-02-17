const prepare_image = (canvas_id, resize_size = 800) => {
    var img = cv.imread(canvas_id);
    var resize_image = new cv.Mat();
    var canvas = document.getElementById(canvas_id);
    var context = canvas.getContext("2d");
    var imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    var array = ndarray(imageData.data, [canvas.height, canvas.width]); // array is a Uint8ClampedArray

    // Get Original Size
    var cols = img.cols;
    var rows = img.rows;

    var resize_ratio = {
        cols: resize_size / cols,
        rows: resize_size / rows
    };

    // Convert from RGBA to RGB

    // Resize
    console.log(`Resize Image into (${resize_size},${resize_size})`);
    //var converted_image = resize_image(array, {targetWidth: resize_size, targetHeight: resize_size, algorithm: 'bicubic'});
    var dsize = new cv.Size(resize_size, resize_size);
    cv.resize (img, resize_image, dsize);

    // Convert to Uint8array
    var converted_image = convertmattoarray(resize_image);

    // Dispose Old Paramter
    img.delete();
    resize_image.delete();

    // Return Image with resize parameter
    return {resize_image: converted_image, orig_size: {cols, rows}, new_size: {cols: resize_size, rows: resize_size}, resize_ratio};
};

const convertmattoarray = (img) => {
    var cols = img.cols;
    var rows = img.rows;
    var channels = 3;

    var img_data = ndarray(new Float32Array(rows * cols * channels), [rows, cols, channels]);
    
    for(var y = 0; y < rows; y++)
        for(var x = 0; x < cols; x++) {
            let pixel = img.ucharPtr(y, x);
            if(x == 0 && y == 0)
                console.log(pixel);
            for(var c = 0; c < channels; c++)
                img_data.set(y, x, c, pixel[c]); 
        }

    return img_data;
};

const getminusparam = (rows, cols) => {
    var img_data = ndarray(new Float32Array(rows * cols * 3), [rows, cols, 3]);

    for(var y = 0; y < rows; y++)
        for(var x = 0; x < cols; x++) {
            img_data.set(y, x, 0, -104);
            img_data.set(y, x, 1, -117);
            img_data.set(y, x, 2, -123); 
        }

    return img_data;
}

const preprocess = (img) => {
    var [img_height, img_width, channels] = img.shape;
    var preprocesed = ndarray(new Float32Array(channels * img_height * img_width), [1, channels, img_height, img_width]);

    // Minus Paramter
    var r = img.pick(null, null, 0),
        g = img.pick(null, null, 1),
        b = img.pick(null, null, 2);

    ndarray.ops.addseq(r, -104);
    ndarray.ops.addseq(g, -117);
    ndarray.ops.addseq(b, -123);

    // Transpose
    ndarray.ops.assign(preprocesed.pick(0, 0, null, null), b);
    ndarray.ops.assign(preprocesed.pick(0, 1, null, null), g);
    ndarray.ops.assign(preprocesed.pick(0, 2, null, null), r);

    console.log('Preprocess');
    console.log(preprocesed.data);

    return preprocesed.data;
};

const product = (x, y) => {
    var size_x = x.length,
        size_y = y.length;
    var result = [];

    for(var i = 0; i < size_x; i++) 
        for(var j = 0; j < size_y; j++) 
            result.push([x[i], y[j]]);

    return result;
};

const range = (num) => {
    var result = [];
    for(var i = 0; i < num; i++)
        result.push(i);

    return result;
};

const PriorBox = (config, image_size) => {
    var min_sizes = config.min_sizes,
        steps = config.steps,
        clip = config.clip,
        image_size = image_size,
        name = "s",
        feature_maps = steps.map((step) => [Math.ceil(image_size[0] / step), Math.ceil(image_size[1] / step)]);
    
    var anchors = [];

    feature_maps.forEach((f, k) => {
        var min_size = min_sizes[k];
        product(range(f[0]), range(f[1])).forEach(([i,j]) => {
            min_size.forEach((m_size) => {
                var s_kx = m_size / image_size[1],
                    s_ky = m_size / image_size[0];
                var dense_cx = [j+0.5].map((x) => x * steps[k] / image_size[1]),
                    dense_cy = [i+0.5].map((y) => y * steps[k] / image_size[0]);
                product(dense_cy, dense_cx).forEach(([cy, cx]) => {
                    anchors.push(cx);
                    anchors.push(cy);
                    anchors.push(s_kx);
                    anchors.push(s_ky);
                })
            });
        });
    });

    var output = ndarray(new Float32Array(anchors), [anchors.length / 4, 4]);

    if(clip)
        output = ndarray.ops.min(1, np.ops.max(output, 0));

    return output;
};

// Decode BBox
const decode_bbox = (bbox, priors, variances) => {
    var loc = ndarray(bbox, [parseInt(bbox.length / 4), 4]);
    console.log(priors);

    var before_prior = priors.hi(null, 2),
        after_prior = priors.lo(null, 2);

    console.log('Before Decode');
    console.log(loc);

    var before_loc = loc.hi(null, 2),
        after_loc = loc.lo(null, 2);

    var before_result = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
    var before_temp = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
    var before_temp2 = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);

    var after_result = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
    var after_temp = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
    var after_temp2 = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
    var after_temp3 = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
    var after_temp4 = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);

    var boxes = ndarray(new Float32Array(before_loc.shape[0] * 4), [before_loc.shape[0], 4]);

    // Before
    ndarray.ops.mul(before_temp, before_loc, after_prior);
    ndarray.ops.muls(before_temp2, before_temp, variances[0]);
    ndarray.ops.add(before_result, before_temp2, before_prior);

    // After
    ndarray.ops.muls(after_temp, after_loc, variances[1]);
    ndarray.ops.exp(after_temp2, after_temp);
    ndarray.ops.mul(after_temp3, after_temp2, after_prior);

    for(var index = 0; index < 4; index++)
        ndarray.ops.assign(after_result.pick(null, index), after_temp3.pick(null, index));

    ndarray.ops.divs(after_temp4, after_temp3, -2);
    ndarray.ops.addeq(before_result, after_temp4);

    ndarray.ops.addeq(after_result, before_result);

    ndarray.ops.assign(boxes.pick(null, 0), before_result.pick(null, 0));
    ndarray.ops.assign(boxes.pick(null, 1), before_result.pick(null, 1));
    ndarray.ops.assign(boxes.pick(null, 2), after_result.pick(null, 0));
    ndarray.ops.assign(boxes.pick(null, 3), after_result.pick(null, 1));

    return boxes;
};

// Decode Landmark
const decode_landm = (landmark, priors, variances) => {
    var landms = ndarray(landmark, [parseInt(landmark.length / 10), 10]);
    var before_prior = priors.hi(null, 2),
        after_prior = priors.lo(null, 2);
    var result = ndarray(new Float32Array(landms.shape[0] * landms.shape[1]), landms.shape);
    var priortemp = ndarray(new Float32Array(after_prior.shape[0] * 2), [after_prior.shape[0], 2]);
    var half_size = parseInt(Math.floor(landms.shape[1] / 2));

    console.log('Before Decode');
    console.log(landms);

    ndarray.ops.muls(priortemp, after_prior, variances[0]);

    for(var index = 0; index < half_size; index++) {
        let temp = ndarray(new Float32Array(landms.shape[0] * 2), [landms.shape[0], 2]);
        let temp2 = ndarray(new Float32Array(landms.shape[0] * 2), [landms.shape[0], 2]);
        let preslice = landms.hi(null, (index + 1)*2).lo(null, index*2);
        ndarray.ops.mul(temp, preslice, priortemp);
        ndarray.ops.add(temp2, temp, before_prior);
        ndarray.ops.assign(result.pick(null, index*2), temp2.pick(null, 0));
        ndarray.ops.assign(result.pick(null, index*2 + 1), temp2.pick(null, 1));
    }
    
    return result;
};

const scale_multiply_bbox = (boxes_arr, scale) => {
    var total_result = boxes_arr.shape[0];
    var boxes_before = ndarray(new Float32Array(total_result * 4), [total_result, 4]);

    for(var index = 0; index < scale.length; index++) {
        let temp = boxes_arr.pick(null,index),
            before_result = ndarray(new Float32Array(total_result), [total_result]);
        ndarray.ops.muls(before_result, temp, scale[index]);
        ndarray.ops.assign(boxes_before.pick(null, index), before_result);
    }

    return boxes_before;
};

const scale_multiply_landms = (landms_arr, scale1) => {
    var total_result = landms_arr.shape[0];
    var landms_before = ndarray(new Float32Array(total_result * 10), [total_result, 10]);

    for(var index = 0; index < scale1.length; index++) {
        let temp = landms_arr.pick(null,index),
            before_landms_result = ndarray(new Float32Array(total_result), [total_result]);
        ndarray.ops.muls(before_landms_result, temp, scale1[index]);
        ndarray.ops.assign(landms_before.pick(null,index), before_landms_result);
    }

    return landms_before;
};

const screen_score = (bbox, scores, landms, threshold) => {
    var total_size = scores.shape[0];
    var index_arr = [];

    for(var index = 0; index < total_size; index++) {
        var score_temp = scores.get(index);
        if(score_temp >= threshold) {
            index_arr.push(index);
        }
    }

    console.log(`Screen Index`);
    console.log(index_arr);

    var result_bbox = ndarray(new Float32Array(index_arr.length * 4), [index_arr.length, 4]);
    var result_scores = ndarray(new Float32Array(index_arr.length), [index_arr.length]);
    var result_landms = ndarray(new Float32Array(index_arr.length * 10), [index_arr.length, 10]);

    index_arr.forEach((index, i) => {
        ndarray.ops.assign(result_bbox.pick(i, null), bbox.pick(index, null));
        ndarray.ops.assign(result_landms.pick(i, null), landms.pick(index, null));
        ndarray.ops.assign(result_scores.pick(i), scores.pick(index));
    });

    console.log([result_bbox, result_scores, result_landms]);

    return [result_bbox, result_scores, result_landms];
};

const sort_score = (bbox, scores, landms, top_k) => {
    var total_size = scores.shape[0];
    var index_sort = new Array(total_size*2);

    for(var index = 0; index < total_size; index++) {
        var temp = scores.get(index);
        index_sort[index] = [index, temp];
    }

    index_sort.sort((a,b) => {
        if(a[1] < b[1]) return 1;
        if(a[1] > b[1]) return -1;

        return 0;
    });

    var max_size = (total_size > top_k) ? top_k : total_size;

    var result_bbox = ndarray(new Float32Array(max_size * 4), [max_size, 4]);
    var result_scores = ndarray(new Float32Array(max_size), [max_size]);
    var result_landms = ndarray(new Float32Array(max_size * 10), [max_size, 10]);

    for(var index = 0; index < max_size; index++) {
        result_scores.set(index, index_sort[index][1]);
        ndarray.ops.assign(result_bbox.pick(index, null), bbox.pick(index_sort[index][0], null)); 
        ndarray.ops.assign(result_landms.pick(index, null), landms.pick(index_sort[index][0], null)); 
    }

    console.log('Sorted Score');
    console.log(result_scores);

    return [result_bbox, result_scores, result_landms];
};

const cpu_nms = (bbox, scores, landms, thresh) => {
    var { max, min } = Math;
    var size = bbox.shape[0];
    var foundLocations = [];
    var pick = [];

    for(var i = 0; i < size; i++) {
        var x1 = bbox.get(i, 0),
            y1 = bbox.get(i, 1),
            x2 = bbox.get(i, 2),
            y2 = bbox.get(i, 3);

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
        console.log('Suppressing');
        var last = foundLocations[0]; //foundLocations.length - 1];
        var suppress = [last];
        pick.push(0); //foundLocations.length - 1);

        for(let i = 1; i < foundLocations.length; i++) {
            const box = foundLocations[i];
            const xx1 = max(box.x1, last.x1);
            const yy1 = max(box.y1, last.y1);
            const xx2 = min(box.x2, last.x2);
            const yy2 = min(box.y2, last.y2);
            const w = max(0, xx2 - xx1 + 1);
            const h = max(0, yy2 - yy1 + 1);
            const overlap = (w*h) / box.area;

            if(overlap >= thresh)
                suppress.push(foundLocations[i]);
        }

        foundLocations = foundLocations.filter((box) => {
            return !suppress.find((supp) => {
                return supp === box;
            })
        });
    }

    var result_bbox = ndarray(new Float32Array(pick.length * 4), [pick.length, 4]);
    var result_scores = ndarray(new Float32Array(pick.length), [pick.length]);
    var result_landms = ndarray(new Float32Array(pick.length * 10), [pick.length, 10]);

    console.log('Pick Index');
    console.log(pick);

    pick.forEach((pick_index, i) => {
        ndarray.ops.assign(result_bbox.pick(i, null), bbox.pick(pick_index, null));
        ndarray.ops.assign(result_scores.pick(i), scores.pick(pick_index));
        ndarray.ops.assign(result_landms.pick(i, null), landms.pick(pick_index, null));
    });

    return [result_bbox, result_scores, result_landms, pick.length];
};

const result_scaling = (bbox, landmark, resize_param) => {
    var size = bbox.shape[0];
    var result_bbox = ndarray(new Float32Array(size * 4), [size, 4]);
    var result_landms = ndarray(new Float32Array(size * 10), [size, 10]);

    // BBox
    for(var i = 0; i < 2; i++) {
        let x = bbox.pick(null, i*2),
            y = bbox.pick(null, i*2 + 1);

        ndarray.ops.divseq(x, resize_param.cols);          // X1 or X2
        ndarray.ops.divseq(y, resize_param.rows);     // Y1 or Y2

        ndarray.ops.assign(result_bbox.pick(null, i*2), x);
        ndarray.ops.assign(result_bbox.pick(null, i*2 + 1), y);
    }

    // Landmark
    for(var i = 0; i < 5; i++) {
        let x = landmark.pick(null, i*2),
            y = landmark.pick(null, i*2 + 1);

        ndarray.ops.divseq(x, resize_param.cols);     // X
        ndarray.ops.divseq(y, resize_param.rows);    // Y

        ndarray.ops.assign(result_landms.pick(null, i*2), x );
        ndarray.ops.assign(result_landms.pick(null, i*2 + 1), y );
    }

    return [result_bbox, result_landms];
};

const drawoutputtocanvas = (output, ctx) => {
    var { bbox , landmark, conf, size} = output; 

    for(var index = 0; index < size; index++) {
        // Bbox
        var x1 = parseInt(bbox.get(index, 0)),
            y1 = parseInt(bbox.get(index, 1)),
            x2 = parseInt(bbox.get(index, 2)),
            y2 = parseInt(bbox.get(index, 3));

        // Draw Bbox
        ctx.beginPath();
        ctx.strokeStyle = '#2ecc71';
        ctx.rect(x1, y1, (x2-x1) + 1, (y2-y1) + 1);
        ctx.stroke();

        // Landmark
        for(var i = 0; i < 5; i++) {
            var x = landmark.get(index, i * 2),
                y = landmark.get(index, i * 2 + 1),
                r = 1,
                sangle = 0,
                eangle = 2 * Math.PI;

            ctx.beginPath();
            ctx.strokeStyle = '#2ecc71';
            ctx.arc(x, y, r, sangle, eangle);
            ctx.stroke();
        }

        // Score
        var score = conf.get(index);
        ctx.font = `lighter 12px sans-serif`;
        ctx.textAlign = "left";
        ctx.textBaseline = 'bottom';
        ctx.fillStyle = '#2ecc71';
        ctx.fillText(`Conf: ${(score*100).toFixed()} %`, x1, y1);
    }
};

const detection = async (img, onnx_session, resize_param, config) => {
    //var new_img = ndarray.pool.clone(img);
    var [img_height, img_width, channels] = img.shape;

    // Scaling
    var scale = [img_width, img_height, img_width, img_height];
    var scale1 = [img_width, img_height, img_width, img_height, 
                  img_width, img_height, img_width, img_height,
                  img_width, img_height];

    // Preprocess
    var inputTensor = new onnx.Tensor(preprocess(img), 'float32', [1, channels, img_width, img_height]);

    // Inference
    var start = new Date().getTime();
    console.log('Inference');
    var output = await onnx_session.run([inputTensor]);
    var outputData = output.values(); //.next().value.data;
    var inference_time = (new Date().getTime() - start)/1000;

    console.log('Inference Complete');
    console.log(`Time usage = ${inference_time} seconds`);

    // Location, Confidence and Landmarks
    var loc = outputData.next().value.data;
    var conf = outputData.next().value.data;
    var landms = outputData.next().value.data;

    var total_result = conf.length / 2;

    // Priorbox
    var priors = PriorBox(config, [img_height, img_width]);

    // Decode
    var boxes_arr = decode_bbox(loc, priors, config.variance);
    var scores_arr = ndarray(conf, [total_result, 2]).pick(null, 1);
    var landms_arr = decode_landm(landms, priors, config.variance);

    var boxes_before = scale_multiply_bbox(boxes_arr, scale);
    var landms_before = scale_multiply_landms(landms_arr, scale1);

    // Screen Low Score
    var [bbox_screen, scores_screen, landms_screen] = screen_score(boxes_before, scores_arr , landms_before, config.confidence_threshold);

    // Keep Top-k
    var [bbox_sorted, scores_sorted, landms_sorted] = sort_score(bbox_screen, scores_screen, landms_screen, config.top_k);

    // NMS
    var [bbox_small, score_result, landms_small, result_size ] = cpu_nms(bbox_sorted, scores_sorted, landms_sorted, config.nms_threshold);
    
    // Debugging Purpose
    //var bbox_small = bbox_sorted;
    //var landms_small = landms_sorted;
    //var result_size = bbox_sorted.shape[0];
    //var score_result = scores_sorted;

    //console.log('After NMS');
    //console.log(bbox_small);

    // Scaling
    var [bbox_result, landms_result] = result_scaling(bbox_small, landms_small, resize_param);

    var output = {
        bbox: bbox_result,
        landmark: landms_result,
        conf: score_result,
        size: result_size
    };

    // Usage Time
    var usage = (new Date().getTime() - start)/1000;    // in seconds.
    console.log(`Finish within ${usage} seconds`);
    console.log('Result');
    console.log(output);

    return output;
};
