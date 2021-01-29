/*
    Convolution Unit
    ===========================
    The function is used for convolutional layer.
*/
class ConvUnit extends tf.layers.Layer {
    constructor(config, nameconv = 'conv', namebn = 'bn', weights = []) { //f, k, s, act = null, name = 'ConvBN') {
        super({});

        this.conv = tf.layers.conv2d({
            filters: config.f,
            kernelSize: config.k,
            strides: config.s, 
            padding:'same',
            dataFormat: 'channelsLast',
            kernelInitializer: 'heNormal',
            kernelRegularizer: 'l1l2',
            use_bias: false,
            name: nameconv
        });

        this.bn = tf.layers.batchNormalization({name : namebn});

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

// Functional Pyramid Network
const FPN = (input, config) => {
    var act = 'relu';
    if(config.out_ch <= 64)
        act = 'lrelu';

    var out_ch = config.out_ch;

    var cnn_output1 = new ConvUnit({f: out_ch, k: 1, s: 1, act:act} , nameconv = 'FPNconv1', namebn = 'FPNbn1');
    var cnn_output2 = new ConvUnit({f: out_ch, k: 1, s: 1, act:act}, nameconv = 'FPNconv2', namebn = 'FPNbn2');
    var cnn_output3 = new ConvUnit({f: out_ch, k: 1, s: 1, act:act}, nameconv = 'FPNconv3', namebn = 'FPNbn3');
    var merge1 = new ConvUnit({f: out_ch, k: 3, s: 1, act:act}, nameconv = 'FPNconv3_1', namebn = 'FPNbn3_1');
    var merge2 = new ConvUnit({f: out_ch, k: 3, s: 1, act:act}, nameconv = 'FPNconv3_2', namebn = 'FPNbn3_2');

    return tf.tidy(() => {
        var output1 = cnn_output1.apply(input[0]);
        var output2 = cnn_output2.apply(input[1]);
        var output3 = cnn_output3.apply(input[2]);

        var up_h = output2.shape[1]; var up_w = output2.shape[2];
        var up3 = tf.image.resizeNearestNeighbor(output3, [up_h, up_w]);
        output2 = output2.add(up3);
        output2 = merge2.apply(output2);
        up3.dispose();

        var up_h = output1.shape[1]; var up_w = output1.shape[2];
        var up2 = tf.image.resizeNearestNeighbor(output2, [up_h, up_w]);
        output1 = output1.add(up2);
        output1 = merge1.apply(output1);
        up2.dispose();

        return [output1, output2, output3];
    });
};

// SSH
const SSH = (input, config, num = 0) => {
    var act = 'relu';
    if(config.out_ch <= 64)
        act = 'lrelu';

    var i = num.toString();
    var out_ch = config.out_ch;
    var cnn_conv_3x3 = new ConvUnit({f: Math.floor(out_ch / 2), k: 3, s: 1, act:null}, nameconv = 'SSHconv4_' + i, namebn = 'SSHbn4_' + i);

    var cnn_conv_5x5_1 = new ConvUnit({f: Math.floor(out_ch / 4), k: 3, s: 1, act:act}, nameconv = 'SSHconv5_' + i, namebn = 'SSHbn5_' + i);
    var cnn_conv_5x5_2 = new ConvUnit({f: Math.floor(out_ch / 4), k: 3, s: 1, act:null}, nameconv = 'SSHconv6_' + i, namebn = 'SSHbn6_' + i);

    var cnn_conv_7x7_2 = new ConvUnit({f: Math.floor(out_ch / 4), k: 3, s: 1, act:act}, nameconv = 'SSHconv7_' + i, namebn = 'SSHbn7_' + i);
    var cnn_conv_7x7_3 = new ConvUnit({f: Math.floor(out_ch / 4), k: 3, s: 1, act:null}, nameconv = 'SSHconv8_' + i, namebn = 'SSHbn8_' + i);

    var relu = tf.layers.reLU();

    return tf.tidy(() => {
        var conv_3x3 = cnn_conv_3x3.apply(input);

        var conv_5x5_1 = cnn_conv_5x5_1.apply(input);
        var conv_5x5 = cnn_conv_5x5_2.apply(conv_5x5_1);

        var conv_7x7_2 = cnn_conv_7x7_2.apply(conv_5x5_1);
        var conv_7x7 = cnn_conv_7x7_3.apply(conv_7x7_2);

        var output = tf.concat([conv_3x3, conv_5x5, conv_7x7], axis = 3);
        output = relu.apply(output);

        return output;
    });
};

const BboxHead = (input, num_anchor, num = 0) => {
    var i = num.toString();

    const conv = tf.layers.conv2d({
        filters: num_anchor * 4,
        kernelSize: 1,
        strides: 1, 
        name: 'Bboxconv_' + i
    });

    //(filters=num_anchor * 4, kernel_size=1, strides=1, name = 'Bboxconv_' + i)
    return tf.tidy(() => {
        h = input.shape[1]; w = input.shape[2];
        x = conv.apply(input);
        return tf.reshape(x, [-1, h * w * num_anchor, 4]);
    });
};

const LandmarkHead = (input, num_anchor, num = 0) => {
    var i = num.toString();

    const conv = tf.layers.conv2d({
        filters: num_anchor * 10,
        kernelSize: 1,
        strides: 1, 
        name: 'Landmconv_' + i
    });

    return tf.tidy(() => {
        h = input.shape[1]; w = input.shape[2];
        x = conv.apply(input);
        return tf.reshape(x, [-1, h * w * num_anchor, 10]);
    });
};

const ClassHead = (input, num_anchor, num = 0) => {
    var i = num.toString();

    const conv = tf.layers.conv2d({
        filters: num_anchor * 2,
        kernelSize: 1,
        strides: 1, 
        name: 'Classconv_' + i
    });

    return tf.tidy(() => {
        h = input.shape[1]; w = input.shape[2];
        x = conv.apply(input);
        return tf.reshape(x, [-1, h * w * num_anchor, 2]);
    });
};

// Meshgrid
const meshgrid_tf = (x,y) => {
    x.print();
    y.print();
    const grid_shape = [y.shape[0], x.shape[0]];
    console.log(grid_shape);
    const grid_x = tf.broadcastTo(tf.reshape(x, [1, -1]), grid_shape);
    const grid_y = tf.broadcastTo(tf.reshape(y, [-1, 1]), grid_shape);
    return [grid_x, grid_y];
};

// Repeat
const repeat_tf = (x, repeat) => {

};

/*
def _meshgrid_tf(x, y):
    """ workaround solution of the tf.meshgrid() issue:
        https://github.com/tensorflow/tensorflow/issues/34470"""
    grid_shape = [tf.shape(y)[0], tf.shape(x)[0]]
    grid_x = tf.broadcast_to(tf.reshape(x, [1, -1]), grid_shape)
    grid_y = tf.broadcast_to(tf.reshape(y, [-1, 1]), grid_shape)
    return grid_x, grid_y
*/

// Prior Box Tensorflow
const prior_box_tf = (image_sizes, min_sizes, steps, clip = false)=> {
    const img_sizes = tf.cast(tf.tensor(image_sizes), 'float32');
    const tf_img_size = tf.reshape(img_sizes, [1, 2])
    const tf_steps = tf.reshape(tf.cast(steps, 'float32'), [-1, 1]);
    const feature_maps = tf.ceil(tf.div(tf_img_size, tf_steps) ).arraySync();

    anchors = []

    for(var k = 0; k < min_sizes.length; k++) {
        const feature_0 = feature_maps[k][0];
        const feature_1 = feature_maps[k][1];
        const [grid_x, grid_y] = meshgrid_tf(tf.range(0, feature_1), tf.range(0, feature_0));
        const cx = tf.div(tf.mul(tf.add(grid_x, tf.scalar(0.5)), tf.scalar(steps[k]) ), tf.scalar(image_sizes[1]) ); //(grid_x + 0.5) * steps[k] / img_sizes[1]; // TODO:
        const cy = tf.div(tf.mul(tf.add(grid_y, tf.scalar(0.5)), tf.scalar(steps[k]) ), tf.scalar(image_sizes[0]) ); //(grid_y + 0.5) * steps[k] / img_sizes[0];
        var cxcy = tf.stack([cx, cy], axis = 1);
        var stacked_cxcy = tf.reshape(cxcy, [-1, 2]);
        var min_sizes_shape = tf.tensor(min_sizes[k]).shape[0];

        console.log('Before Repeat');
        console.log(min_sizes_shape);
        stacked_cxcy.print();
        var repeatcxcy = tf.repeat(stacked_cxcy, repeat=min_sizes_shape, axis = 0);

        repeatcxcy.print();
    }
};

/*
"""prior box"""
    image_sizes = tf.cast(tf.convert_to_tensor(image_sizes), tf.float32)
    feature_maps = tf.math.ceil(
        tf.reshape(image_sizes, [1, 2]) /
        tf.reshape(tf.cast(steps, tf.float32), [-1, 1]))

    anchors = []
    for k in range(len(min_sizes)):
        grid_x, grid_y = _meshgrid_tf(tf.range(feature_maps[k][1]),
                                      tf.range(feature_maps[k][0]))
        cx = (grid_x + 0.5) * steps[k] / image_sizes[1]
        cy = (grid_y + 0.5) * steps[k] / image_sizes[0]
        cxcy = tf.stack([cx, cy], axis=-1)
        cxcy = tf.reshape(cxcy, [-1, 2])
        cxcy = tf.repeat(cxcy, repeats=tf.shape(min_sizes[k])[0], axis=0)

        sx = min_sizes[k] / image_sizes[1]
        sy = min_sizes[k] / image_sizes[0]
        sxsy = tf.stack([sx, sy], 1)
        sxsy = tf.repeat(sxsy[tf.newaxis],
                         repeats=tf.shape(grid_x)[0] * tf.shape(grid_x)[1],
                         axis=0)
        sxsy = tf.reshape(sxsy, [-1, 2])

        anchors.append(tf.concat([cxcy, sxsy], 1))

    output = tf.concat(anchors, axis=0)

    if clip:
        output = tf.clip_by_value(output, 0, 1)

    return output
*/

// RetinaFaceModel
const detection = (tensor, resnet_backbone, config) => {
    
    const output = resnet_backbone.predict(tensor, { batchSize: 1 });
    console.log('Backbone Model predict successfully.');

    const fpn_output = FPN(output, config);
    console.log('FPN predict successfully.');

    const features = fpn_output.map((f, i) => SSH(f, config, i));
    console.log('SSH predict successfully.');

    const num_anchor = config.min_size[0].length;
    const bbox_regressions = tf.concat(features.map((f, i) => BboxHead(f, num_anchor, i)), axis = 1);
    const landm_regressions = tf.concat(features.map((f, i) => LandmarkHead(f, num_anchor, i)), axis = 1);
    const classifications = tf.concat(features.map((f, i) => ClassHead(f, num_anchor, i)), axis = 1);
    const classify_result = classifications.softmax();
    const total_result = classify_result.shape[1];  // 10k
    
    console.log('Regression and Classification Successfully.');
    console.log(`Bounding Box Shape = ${bbox_regressions.shape}`);
    console.log(`Landmark Shape = ${landm_regressions.shape}`);
    console.log(`Classify Shape = ${classify_result.shape}`);

    // Show Result
    const first_arr = classify_result.slice([0,0,0], [1, -1, 1]);
    const second_arr = classify_result.slice([0,0,1], [1, -1, 1]);

    const preds = tf.concat([bbox_regressions, landm_regressions, tf.onesLike(first_arr) , second_arr], axis = 2);
    const preds_result = preds.reshape([ total_result, preds.shape[2] ]);
    preds.dispose();

    console.log('Predictions');
    console.log(preds_result);
    console.log(preds_result.shape);

    priors = prior_box_tf([tensor.shape[1], tensor.shape[2]], config.min_size, config.steps, config.clip);

    // Dispose
    preds.dispose();
    features.dispose();
    bbox_regressions.dispose();
    landm_regressions.dispose();
    classifications.dispose();
    classify_result.dispose();
    first_arr.dispose(); second_arr.dispose();
    preds.dispose();
};