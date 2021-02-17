/* 
    Load Weight
*/
const load_json = (json) => new Promise((resolve, reject) => {
    var xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    try {
        xobj.open('GET',json, true); // Replace 'my_data' with the path to your file
        xobj.onreadystatechange = function () {
            if (xobj.readyState == 4 && xobj.status == "200") {
            // Required use of an anonymous callback as .open will NOT return a value but simply returns undefined in asynchronous mode
                var data = JSON.parse(xobj.responseText);
                var obj = {}
                
                for(var index = 0; index < data.length; index++) {
                    var result = data[index];
                    obj[result.name] = result.weight;
                }

                resolve(obj);
            }
        };
        xobj.send(null);
    } catch(err) {
        reject(err);
    }
});

const load_weight = async (weight) => {
    const data = await load_json(`${weight}`);
    return data;
};

const find_weight = (weight, name) => {
    const data = weight.find(element => element.name == name);

    return data;
}
/*
    Batchnormalizatiomn
*/
class BatchNormalization extends tf.layers.batchNormalization {
    constructor(config, axis=-1, momentum=0.9, epsilon=1e-5, center=true, scale=true, name='BNlayer', weights_bn = []) {
        super({
            ...config,
            axis : axis,
            momentum: momentum,
            epsilon: epsilon,
            center: center,
            scale: scale,
            weights: weights_bn,
            name: name
        });
    }

    call(input, training = false) {
        return tf.tidy(() => {
            
            var train = tf.scalar(training);

            if(training == null)
                train = tf.scalar(false)
            
            var training = train && this.trainable;
    
            return this.apply(input, training);
        });
        
    }

    static get className() {
        return 'BN';
    }
}
/*
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                 scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)

        return super().call(x, training)

*/

/* L2 regularizer */
class L2 {

    static className = 'L2';

    constructor(config) {
       return tf.regularizers.l1l2(config)
    }
}

/*
    Convolution Unit
    ===========================
    The function is used for convolutional layer.
*/
class ConvUnit extends tf.layers.Layer {
    constructor(config, nameconv = 'conv', namebn = 'bn') { //}, weights = []) { //f, k, s, act = null, name = 'ConvBN') {
        super(config);
        var weight = config.weight;

        var temp_weight = weight[nameconv][0].weight;
        var conv_weight = tf.tensor(temp_weight);
        var bias_conv = tf.zeros([config.f]);

        this.conv_weights = [conv_weight, bias_conv];

        this.conv = tf.layers.conv2d({
            filters: config.f,
            kernelSize: config.k,
            strides: config.s, 
            padding:'same',
            dataFormat: 'channelsLast',
            kernelInitializer: 'heNormal',
            kernelRegularizer: 'L2',
            use_bias: false,
            name: nameconv,
            weights: this.conv_weights
        });

        var bn_weight = weight[namebn];

        this.weights_bn = [];

        for(var index = 0; index < bn_weight.length; index++)
            this.weights_bn.push(tf.tensor(bn_weight[index].weight));

        this.bn = new BatchNormalization(config, -1, 0.9, 1e-5, true, true, namebn, this.weights_bn);

        if (config.act == null)
            this.activation = null;
        else if (config.act = 'relu')
            this.activation = tf.layers.reLU();
        else if (config.act == 'lrelu')
            this.activation = tf.layers.leakyReLU({alpha : 0.1});
    }

    call(input) {
        return tf.tidy(() => {
            const result = this.bn.apply(this.conv.apply(input));

            if(this.activation) {
                return this.activation.apply(result);
            } else
                return result;
        });
    }

    static get className() {
        return 'ConvBN';
    }
}


const Backbone = (predictor) => {
    const network = predictor;
    
    const backbone = (input) => {
        const output = network.predict(input);
        return tf.model({
            input: input,
            output: output
        })
    };
    
    return backbone;
};

class FPN extends tf.layers.Layer {
    constructor(config) {
        super(config);
        var out_ch = config.out_ch;
        var act = 'relu';
        if(out_ch <= 64)
            act = 'lrelu';

        this.output1 = new ConvUnit({f: out_ch, k: 1, s: 1, act:act, weight: config.weight} , 'FPNconv1',  'FPNbn1');
        this.output2 = new ConvUnit({f: out_ch, k: 1, s: 1, act:act, weight: config.weight}, 'FPNconv2',  'FPNbn2');
        this.output3 = new ConvUnit({f: out_ch, k: 1, s: 1, act:act, weight: config.weight}, 'FPNconv3', 'FPNbn3');
        this.merge1 = new ConvUnit({f: out_ch, k: 3, s: 1, act:act, weight: config.weight}, 'FPNconv3_1', 'FPNbn3_1');
        this.merge2 = new ConvUnit({f: out_ch, k: 3, s: 1, act:act, weight: config.weight}, 'FPNconv3_2', 'FPNbn3_2');     
    
    }

    call(input) {
        return tf.tidy(() => {
            console.log('FPN');
            var output1 = this.output1.apply(input[0]);
            var output2 = this.output2.apply(input[1]);
            var output3 = this.output3.apply(input[2]);
    
            var up_h = output2.shape[1]; var up_w = output2.shape[2];
            var up3 = tf.image.resizeNearestNeighbor(output3, [up_h, up_w]);
            output2 = output2.add(up3);
            output2 = this.merge2.apply(output2);
            up3.dispose();
    
            var up_h = output1.shape[1]; var up_w = output1.shape[2];
            var up2 = tf.image.resizeNearestNeighbor(output2, [up_h, up_w]);
            output1 = output1.add(up2);
            output1 = this.merge1.apply(output1);
            up2.dispose();
    
            return [output1, output2, output3];
        });
    }

    static get className() {
        return 'FPN';
    }
}

class SSH extends tf.layers.Layer {
    constructor(config) {
        super(config);

        var out_ch = config.out_ch;
        var act = 'relu';
        if(out_ch <= 64)
            act = 'lrelu';
    
        var i = config.num.toString();

        this.conv_3x3 = new ConvUnit({f: Math.floor(out_ch / 2), k: 3, s: 1, act:null, weight: config.weight}, 'SSHconv4_' + i, 'SSHbn4_' + i);
    
        this.conv_5x5_1 = new ConvUnit({f: Math.floor(out_ch / 4), k: 3, s: 1, act:act, weight: config.weight}, 'SSHconv5_' + i, 'SSHbn5_' + i);
        this.conv_5x5_2 = new ConvUnit({f: Math.floor(out_ch / 4), k: 3, s: 1, act:null, weight: config.weight}, 'SSHconv6_' + i, 'SSHbn6_' + i);
    
        this.conv_7x7_2 = new ConvUnit({f: Math.floor(out_ch / 4), k: 3, s: 1, act:act, weight: config.weight}, 'SSHconv7_' + i, 'SSHbn7_' + i);
        this.conv_7x7_3 = new ConvUnit({f: Math.floor(out_ch / 4), k: 3, s: 1, act:null, weight: config.weight}, 'SSHconv8_' + i, 'SSHbn8_' + i);
    
        this.relu = tf.layers.reLU();    
    }

    call(input) {
        return tf.tidy(() => {
            console.log('SSH');
            var conv_3x3 = this.conv_3x3.apply(input);

            var conv_5x5_1 = this.conv_5x5_1.apply(input);
            var conv_5x5 = this.conv_5x5_2.apply(conv_5x5_1);

            var conv_7x7_2 = this.conv_7x7_2.apply(conv_5x5_1);
            var conv_7x7 = this.conv_7x7_3.apply(conv_7x7_2);

            var output = tf.concat([conv_3x3, conv_5x5, conv_7x7], 3);

            output = this.relu.apply(output);

            return output;
        });    

    }

    static get className() {
        return 'SSH';
    }
}

class BboxHead extends tf.layers.Layer {
    constructor(config) {
        super(config);
        var i = config.num.toString();
        var weight = config.weight;

        this.num_anchor = config.numAnchor;

        const name = 'Bboxconv_' + i;
        
        this.conv = tf.layers.conv2d({
            filters: this.num_anchor * 4,
            kernelSize: 1,
            strides: 1, 
            name: name,
            weights: [tf.tensor(weight[name][0].weight), tf.tensor(weight[name][1].weight)]
        });        
    }

    call (input) {
        return tf.tidy(() => {
            console.log('BBoxHead');
            var h = input.shape[1]; var w = input.shape[2];
            var x = this.conv.apply(input);
            return tf.reshape(x, [-1, h * w * this.num_anchor, 4]);
        });
    }

    static get className() {
        return 'BboxHead';
    }
}

class LandmarkHead extends tf.layers.Layer {
    constructor(config) {
        super(config);
        var i = config.num.toString();
        var weight = config.weight;

        this.num_anchor = config.numAnchor;

        const name = 'Landmconv_' + i;
        
        this.conv = tf.layers.conv2d({
            filters: this.num_anchor * 10,
            kernelSize: 1,
            strides: 1, 
            name: name,
            weights: [tf.tensor(weight[name][0].weight), tf.tensor(weight[name][1].weight)]
        });        
    }

    call (input) {
        return tf.tidy(() => {
            console.log('LandmarkHead');
            var h = input.shape[1]; var w = input.shape[2];
            var x = this.conv.apply(input);
            return tf.reshape(x, [-1, h * w * this.num_anchor, 10]);
        });
    }

    static get className() {
        return 'LandmarkHead';
    }
}

class ClassHead extends tf.layers.Layer {
    constructor(config) {
        super(config);
        var weight = config.weight;
        var i = config.num.toString();
        this.num_anchor = config.numAnchor;
        const name = 'Classconv_' + i;
        
        this.conv = tf.layers.conv2d({
            filters: this.num_anchor * 2,
            kernelSize: 1,
            strides: 1, 
            name: name,
            weights: [tf.tensor(weight[name][0].weight), tf.tensor(weight[name][1].weight)]
        });        
    }

    call (input) {
        return tf.tidy(() => {
            console.log('ClassHead');
            var h = input.shape[1]; var w = input.shape[2];
            var x = this.conv.apply(input);
            return tf.reshape(x, [-1, h * w * this.num_anchor, 2]);
        });
    }

    static get className() {
        return 'ClassHead';
    }
}

tf.serialization.registerClass(ConvUnit);
tf.serialization.registerClass(FPN);
tf.serialization.registerClass(SSH);
tf.serialization.registerClass(BboxHead);
tf.serialization.registerClass(LandmarkHead);
tf.serialization.registerClass(ClassHead);
tf.serialization.registerClass(L2);
tf.serialization.registerClass(BatchNormalization);

// Meshgrid
const meshgrid_tf = (x,y) => {
    const grid_shape = [y.shape[0], x.shape[0]];
    const grid_x = tf.broadcastTo(tf.reshape(x, [1, -1]), grid_shape);
    const grid_y = tf.broadcastTo(tf.reshape(y, [-1, 1]), grid_shape);
    return [grid_x, grid_y];
};

// Repeat
// Works only axis = 0;
const repeat_tf = async (x, repeat) => {
    const arr_x = await x.array();
    const size_x = arr_x.length;
    var result = [];

    for(var index = 0; index < size_x; index++) {
        var data = arr_x[index];

        for(var rep = 0; rep < repeat; rep++)
            result.push(data);
    }

    return tf.tensor(result);
};

// Prior Box Tensorflow
const prior_box_tf = async (image_sizes, min_sizes, steps, clip = false)=> {
    const img_sizes = tf.cast(tf.tensor(image_sizes), 'float32');

    const tf_img_size = tf.reshape(img_sizes, [1, 2])

    const tf_steps = tf.reshape(tf.cast(tf.tensor(steps), 'float32'), [-1, 1]);
    const feature_maps = await tf.ceil(tf.div(tf_img_size, tf_steps) ).array();

    var anchors = []
    var output = null;

    for(var k = 0; k < min_sizes.length; k++) {
        const feature_0 = feature_maps[k][0];
        const feature_1 = feature_maps[k][1];
        const [grid_x, grid_y] = meshgrid_tf(tf.range(0, feature_1), tf.range(0, feature_0));

        const cx = tf.div(tf.mul(tf.add(grid_x, tf.scalar(0.5)), tf.scalar(steps[k]) ), (img_sizes.dataSync())[1]);
        const cy = tf.div(tf.mul(tf.add(grid_y, tf.scalar(0.5)), tf.scalar(steps[k]) ), (img_sizes.dataSync())[0]);

        var stacked_cxcy = tf.stack([cx, cy], axis = -1);
        stacked_cxcy = tf.reshape(stacked_cxcy, [-1, 2]);

        const min_sizes_shape = tf.tensor(min_sizes[k]).shape[0];
        var cxcy = await repeat_tf(stacked_cxcy, repeat = min_sizes_shape);

        const sx = tf.div(tf.tensor(min_sizes[k]), (img_sizes.dataSync())[1]);
        const sy = tf.div(tf.tensor(min_sizes[k]), (img_sizes.dataSync())[0]);
        const stacked_sxsy = tf.stack([sx, sy], 1);

        const repeatsxsy = await repeat_tf(stacked_sxsy.expandDims(), grid_x.shape[0] * grid_x.shape[1]);
        var sxsy = tf.reshape(repeatsxsy, [-1, 2]);

        anchors.push(tf.concat([cxcy, sxsy], 1));
    }

    output = tf.concat(anchors, axis = 0);

    if(clip)
        output = tf.clipByValue(output, 0, 1);

    return output;
};

const decode_tf = (pred, priors, variances) => {
    const bbox_labels = pred.slice([0,0],[-1, 4]);
    const bbox = decode_bbox_result(bbox_labels, priors, variances);

    const landmark_labels = pred.slice([0,4],[-1,10]);
    const landm = decode_landm_result(landmark_labels, priors, variances);

    const landm_valid = pred.slice([0,14],[-1,1]);
    const conf = pred.slice([0,15],[-1,1]);

    console.log('Decoding');
    console.log(bbox.dataSync());
    console.log(landm.dataSync());
    console.log(conf.dataSync());

    return tf.concat([bbox, landm, landm_valid, conf], axis = 1); //, axis = 1);
};

const decode_bbox_result = (pred, priors, variances) => {
    var before_priors = priors.slice([0,0],[-1,2]),
        after_priors = priors.slice([0,2],[-1,2]);
    var before_pred = pred.slice([0,0],[-1,2]),
        after_pred = pred.slice([0,2],[-1,-1]);

    // Center
    var centers = tf.add(before_priors, tf.mul(before_pred, tf.mul(after_priors, tf.scalar(variances[0]))));
    
    // Side
    var sides = tf.mul(after_priors, tf.exp(tf.mul(after_pred, tf.scalar(variances[1])) ));

    // Division
    var post = tf.div(sides, tf.scalar(2));

    // BBox
    var bbox =  tf.concat([tf.sub(centers, post), tf.add(centers, post)], axis = 1);

    return bbox;
};

const decode_landm_result = (pred, priors, variances = [0.1, 0.2]) => {
    var before_priors = priors.slice([0,0],[-1,2]),
        after_priors = priors.slice([0,2],[-1,2]);
    var variances_prior = tf.mul(after_priors, tf.scalar(variances[0]));
    var landm = [];

    for(var i = 0; i < 5; i++) {
        let temp = tf.mul( pred.slice([0,2*i], [-1,2]), variances_prior);
        let result = tf.add(before_priors, temp);
        landm.push(result);
    }

    return tf.concat(landm, axis = 1);
};

const convert_bbox = (bbox) => {
    // Convert into y1,x1 / y2,x2
    var x1 = bbox.slice([0,0],[-1,1]),
        y1 = bbox.slice([0,1],[-1,1]),
        x2 = bbox.slice([0,2],[-1,1]),
        y2 = bbox.slice([0,3],[-1,1]);

    return tf.concat([y1,x1,y2,x2], axis = 1);
};

// RetinaFaceModel
const tensorflow_detection = async (input, resnet_backbone, weight_id, config) => {
    const backbone = resnet_backbone.predict(input);
    
    // Weight
    var weight = config.weight[weight_id];
    var weights = await load_json(weight);

    console.log('Weight');
    console.log(weights);
    
    var FPNlayer = new FPN({...config, name: "FPN", weight: weights});
    //FPNlayer.setWeights(fpn_weight);
    var SSHlayer = [];
    var BboxLayer = [];
    var LandmarkLayer = [];
    var ClassLayer = [];
    var num_anchor = config.min_size[0].length;

    for(var i = 0; i < 3; i++) {
        SSHlayer.push(new SSH({...config, name: 'SSH', num: i, weight: weights}) );
        BboxLayer.push(new BboxHead({...config, numAnchor: num_anchor, num: i, weight: weights}) );
        LandmarkLayer.push(new LandmarkHead({...config, numAnchor: num_anchor, num: i, weight: weights}) );
        ClassLayer.push(new ClassHead({...config, numAnchor: num_anchor, num: i, weight: weights}) );
    }

    const FPNresult = FPNlayer.apply(backbone);

    const features = FPNresult.map((f, i) => SSHlayer[i].apply(f) );
    const bbox_regressions = tf.concat(features.map((f, i) => BboxLayer[i].apply(f) ), axis = 1);
    const landm_regressions = tf.concat(features.map((f, i) => LandmarkLayer[i].apply(f) ), axis = 1);
    const classifications = tf.concat(features.map((f, i) => ClassLayer[i].apply(f) ), axis = 1);
    const classify_result = classifications.softmax();
    const total_result = classify_result.shape[1];

    var first_arr = classify_result.slice([0,0,0], [1, -1, 1]);
    var second_arr = classify_result.slice([0,0,1], [1, -1, 1]);

    var preds = tf.concat([bbox_regressions, landm_regressions, tf.onesLike(first_arr) , second_arr], axis = 2);
    var preds_result = preds.reshape([ total_result, preds.shape[2] ]);
    preds.dispose();

    var priors = await prior_box_tf([input.shape[2], input.shape[1]], config.min_size, config.steps, config.clip);
    var decode_preds = await decode_tf(preds_result, priors, config.variances);
    var boxes_before = convert_bbox(decode_preds.slice([0,0],[-1, 4]));
    var scores_before = decode_preds.slice([0,15],[-1,1]);
    //var landms_before = decode_preds.slice([0,4],[-1,10]);

    // NMS
    var selected_indices = await tf.image.nonMaxSuppressionAsync(boxes_before, scores_before.squeeze(), decode_preds.shape[0], config.iou_thresh, config.score_thresh);
    var output = tf.gather(decode_preds, selected_indices);

    // Dispose
    preds.dispose();
    bbox_regressions.dispose();
    landm_regressions.dispose();
    classifications.dispose();
    classify_result.dispose();
    first_arr.dispose(); second_arr.dispose();
    preds.dispose();
    selected_indices.dispose();

    return output;
};