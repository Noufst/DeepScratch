//require('import-export');
require("babel-core/register");
require("babel-polyfill");

const tf = require('@tensorflow/tfjs');
const iris = require('./training.json');
const irisTesting = require('./testing.json');

// MNIST data
let data_dir = require('./data/data.js');
let MnistData = data_dir.MnistData;
const IMAGE_H = 28;
const IMAGE_W = 28;

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Cast = require('../../util/cast');
const log = require('../../util/log');
const formatMessage = require('format-message');
const RenderedTarget = require('../../sprites/rendered-target');
const StageLayering = require('../../engine/stage-layering');
const cocoSsd = require('@tensorflow-models/coco-ssd');

const EPOCHS_DEFAULT = 10;
const LAYERS_DEFAULT = 2;
const BATCH_SIZE_DEFAULT = 320;

const Message = {
    video_toggle: {
        'en': 'turn video [VIDEO_STATE]',
        'ar': 'تشغيل الفديو [VIDEO_STATE]'
    },
    on: {
        'en': 'on',
        'ar': 'تشغيل'
    },
    off: {
        'en': 'off',
        'ar': 'ايقاف'
    },
    video_on_flipped: {
        'en': 'on flipped',
        'ar': 'مقلوبة'
    }
}

const AvailableLocales = ['en', 'ar'];

let cnn_model = null;
let dense_model = null;
let rnn_model = null;
let epochs = EPOCHS_DEFAULT;
let accuracy = "";
let training_accuracy = "";
let loss = "";
let prediction = "";

class Scratch3DeepScratch {

    constructor(runtime) {
        this.runtime = runtime;
        this.video = document.createElement("video");
        this.video.width = 408;
        this.video.height = 306;
        this.video.autoplay = true;
        this.video.style.display = "none";
        this.canvas = document.createElement('canvas');

        // Load the pre-trained model
        this.interval = 1000;
        this.coordinate = {
            text: "1",
            value: "no object detected"
        }
        this.objectClasses = [
            {
                text: "1",
                value: "no object detected"
            },
            {
                text: "2",
                value: "nno object detected"
            },
            {
                text: "3",
                value: "no object detected"
            }
        ];
        this.objectScores = [{
            text: "1",
            value: "no object detected"
        },
        {
            text: "2",
            value: "no object detected"
        },
        {
            text: "3",
            value: "no object detected"
        }];
        this.objectXPos = [{
            text: "1",
            value: 0
        },
        {
            text: "2",
            value: 0
        },
        {
            text: "3",
            value: 0
        }];
        this.objectYPos = [{
            text: "1",
            value: 0
        },
        {
            text: "2",
            value: 0
        },
        {
            text: "3",
            value: 0
        }];
        const sp = this.runtime.getTargetForStage().getCostumes();
        const targets = this.runtime.executableTargets;
        var i;
        for (i = 0; i < targets.length; i++) {
            console.log(targets[i]);
            console.log(targets[i].sprite);
            if (targets[i].isStage && targets[i].sprite) {
                this.test = targets[i];
            }
        }
        this.locale = this.setLocale();

    }

    getInfo() {

        return {
            id: 'deepscratch',
            color1: '#1ebdad',
            color2: '#0aab9b',
            name: 'Deep Scratch',
            blocks: [
                {
                    opcode: 'setEpochs',
                    blockType: BlockType.COMMAND,
                    text: 'set number of epochs [Number]',
                    arguments: {
                        Number: {
                            type: ArgumentType.NUMBER,
                            defaultValue: EPOCHS_DEFAULT
                        }
                    }
                },
                '---',
                {
                    opcode: 'dense_train',
                    blockType: BlockType.COMMAND,
                    text: 'train Dense: data [Data] #layers [Layers]',
                    arguments: {
                        Data: {
                            type: ArgumentType.STRING,
                            menu: 'dense_data_menu'
                        },
                        Layers: {
                            type: ArgumentType.STRING,
                            defaultValue: LAYERS_DEFAULT
                        }
                    }
                },
                {
                    opcode: 'RNN_train',
                    blockType: BlockType.COMMAND,
                    text: 'train RNN: data [Data]',
                    arguments: {
                        Data: {
                            type: ArgumentType.STRING,
                            menu: 'rnn_data_menu'
                        }
                    }
                },
                {
                    opcode: 'CNN_train',
                    blockType: BlockType.COMMAND,
                    text: 'train CNN: data [Data] batch size [BS]',
                    arguments: {
                        Data: {
                            type: ArgumentType.STRING,
                            menu: "cnn_data_menu"
                        },
                        BS: {
                            type: ArgumentType.NUMBER,
                            defaultValue: BATCH_SIZE_DEFAULT
                        }
                    }
                },
                '---',
                {
                    opcode: 'predictIris',
                    blockType: BlockType.COMMAND,
                    text: 'predict iris: sepal length [sepal_length] width [sepal_width] petal length [petal_length] width [petal_width]',
                    arguments: {
                        sepal_length: {
                            type: ArgumentType.NUMBER
                        },
                        sepal_width: {
                            type: ArgumentType.NUMBER
                        },
                        petal_length: {
                            type: ArgumentType.NUMBER
                        },
                        petal_width: {
                            type: ArgumentType.NUMBER
                        }
                    }
                },
                {
                    opcode: 'predictMNIST',
                    blockType: BlockType.COMMAND,
                    text: 'predict MNIST',
                },
                '---',
                {
                    opcode: 'getAccuracy',
                    blockType: BlockType.REPORTER,
                    text: 'testing accuracy',
                    arguments: {
                    }
                },
                {
                    opcode: 'getTrainingAccuracy',
                    blockType: BlockType.REPORTER,
                    text: 'training accuracy',
                    arguments: {
                    }
                },
                {
                    opcode: 'getLoss',
                    blockType: BlockType.REPORTER,
                    text: 'loss',
                    arguments: {
                    }
                },
                {
                    opcode: 'getPrediction',
                    blockType: BlockType.REPORTER,
                    text: 'prediction',
                    arguments: {
                    }
                }, 
                '---',
                // Pre-trained model
                {
                    opcode: 'videoToggle',
                    text: Message.video_toggle[this.locale],
                    blockType: BlockType.COMMAND,
                    arguments: {
                        VIDEO_STATE: {
                            type: ArgumentType.STRING,
                            menu: 'video_menu',
                            defaultValue: 'off'
                        }
                    }
                },
                {
                    opcode: 'detectObj',
                    blockType: BlockType.COMMAND,
                    text: 'detect objects'
                    // arguments: {
                    //     OBJNO: {
                    //         type: ArgumentType.NUMBER,
                    //         menu: 'obj_menu',
                    //         defaultValue: 1
                    //     }
                    // }
                },
                {
                    opcode: 'objectClass',
                    blockType: BlockType.REPORTER,
                    text: 'object Class [OBNO]',
                    arguments: {
                        OBNO: {
                            type: ArgumentType.STRING,
                            menu: 'classes_menu',
                            defaultValue: "1"
                        }
                    }
                },
                {
                    opcode: 'objectScore',
                    blockType: BlockType.REPORTER,
                    text: 'object score [OBSCSCORES]',
                    arguments: {
                        OBSCSCORES: {
                            type: ArgumentType.STRING,
                            menu: 'scores_menu',
                            defaultValue: '1'
                        }
                    }
                },
                {
                    opcode: 'objectxPos',
                    blockType: BlockType.REPORTER,
                    text: 'object x position [OBX]',
                    arguments: {
                        OBX: {
                            type: ArgumentType.NUMBER,
                            menu: 'posx_menu',
                            defaultValue: '1'
                        }
                    }
                },
                {
                    opcode: 'objectyPos',
                    blockType: BlockType.REPORTER,
                    text: 'object y position [OBY]',
                    arguments: {
                        OBY: {
                            type: ArgumentType.NUMBER,
                            menu: 'posy_menu',
                            defaultValue: '1'
                        }
                    }
                }
            ],
            menus: {
                dense_data_menu: ['iris'],
                cnn_data_menu: ["MNIST"],
                rnn_data_menu: ['iris'],
                video_menu: this.getVideoMenu(),
                classes_menu: this.getCLassMenu(),
                scores_menu: this.getScoreMenu(),
                posx_menu: this.getXPosMenu(),
                posy_menu: this.getYPosMenu()
            }
        };
    } // End getinfo

    getAccuracy() {
        //console.log(this.accuracy);
        return accuracy;
    }

    getTrainingAccuracy() {
        return training_accuracy;
    }

    getLoss() {
        //console.log(this.loss);
        return loss;
    }

    getPrediction() {
        return prediction;
    }

    setEpochs(args) {
        epochs = args.Number;
    }

    reset_values() {
        accuracy = "";
        training_accuracy = "";
        loss = "";
        prediction = "";
        dense_model = null;
        rnn_model = null;
        cnn_model = null;
    }

    // Load iris data
    load_iris() {

        // Mapping the trainingdata
        let trainingData = tf.tensor2d(iris.map(item => [
            item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]), [130, 4])

        // Mapping the testing data
        let testingData = tf.tensor2d(irisTesting.map(item => [
            item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]), [14, 4])

        // Mapping the output data
        const outputData = tf.tensor2d(iris.map(item => [
            item.species === 'setosa' ? 1 : 0,
            item.species === 'virginica' ? 1 : 0,
            item.species === 'versicolor' ? 1 : 0]), [130, 3])

        // Mapping the output data
        const testingOutputData = tf.tensor2d(irisTesting.map(item => [
            item.species === 'setosa' ? 1 : 0,
            item.species === 'virginica' ? 1 : 0,
            item.species === 'versicolor' ? 1 : 0]), [14, 3])

        return [trainingData, testingData, outputData, testingOutputData]

    }

    //_____________________Dense train_____________________________________
    dense_train(args) {

        this.reset_values();

        // ___________________________IRIS data ____________________________
        if (args.Data == "iris") {

            const iris_data = this.load_iris();
            const trainingData = iris_data[0];
            const testingData = iris_data[1];
            const outputData = iris_data[2];
            const testingOutputData = iris_data[3];

            // Creating Model
            dense_model = tf.sequential();

            dense_model.add(tf.layers.dense(
                {
                    inputShape: 4,
                    activation: 'sigmoid',
                    units: 5
                }
            ));

            for (let i = 1; i <= args.Layers; i++) {
                dense_model.add(tf.layers.dense(
                    {
                        inputShape: 5,
                        activation: 'sigmoid',
                        units: 5
                    }
                ));
            }

            dense_model.add(tf.layers.dense(
                {
                    inputShape: 5,
                    units: 3,
                    activation: 'softmax'
                }
            ));


            console.log(dense_model.summary());
            // compiling model
            dense_model.compile({
                loss: "categoricalCrossentropy",
                metrics: ["accuracy"],
                optimizer: tf.train.adam(.06)
            })

            dense_model.fit(trainingData, outputData, {
                epochs: epochs, shuffle: false, callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        //console.log("EPOCH END");
                        console.log("Loss: ", logs.loss);
                        loss = logs.loss;
                        console.log("Training Accuracy: ", logs.acc);
                        training_accuracy = logs.acc;

                    }
                }
            }).then(
                (history) => {
                    const evaluation = dense_model.evaluate(testingData, testingOutputData)
                    // // Loss
                    // console.log("Testing Loss: ");
                    // console.log(evaluation[0].print());

                    // Testing Accuracy
                    console.log("Testing Accuracy: ");
                    evaluation[1].array().then(array => {
                        accuracy = array;
                        console.log(this.accuracy);
                    });

                    console.log("DONE");
                });

        } // End iris data

        epochs = EPOCHS_DEFAULT;
        layers = LAYERS_DEFAULT;

    }

    //____________________ Predict iris  _____________________________
    predictIris(args) {

        prediction = "";

        if (dense_model != null) {

            let newData = tf.tensor2d([[Number(args.sepal_length), Number(args.sepal_width), Number(args.petal_length), Number(args.petal_width)]]);

            dense_model.predict(newData).array().then(array => {

                array = array[0];
                let max = array[0]
                if (array[1] > max) {
                    max = array[1];
                }
                if (array[2] > max) {
                    max = array[2];
                }

                if (max == array[0]) {
                    prediction = "setosa";
                } else if (max == array[1]) {
                    prediction = "virginica";
                } else {
                    prediction = "versicolor";
                }
            });

        } else if (rnn_model != null) {

            let newData = tf.tensor2d([[Number(args.sepal_length), Number(args.sepal_width), Number(args.petal_length), Number(args.petal_width)]]);

            // Reshape input for RNN
            newData = newData.reshape([1, 1, 4]);    

            rnn_model.predict(newData).array().then(array => {

                array = array[0];
                let max = array[0]
                if (array[1] > max) {
                    max = array[1];
                }
                if (array[2] > max) {
                    max = array[2];
                }

                if (max == array[0]) {
                    prediction = "setosa";
                } else if (max == array[1]) {
                    prediction = "virginica";
                } else {
                    prediction = "versicolor";
                }
            });

        }
    }

    //_____________________RNN train_____________________________________
    RNN_train(args) {

        this.reset_values();

        // ___________________________IRIS data ____________________________
        if (args.Data == "iris") {
            const iris_data = this.load_iris();
            let trainingData = iris_data[0];
            let testingData = iris_data[1];
            let outputData = iris_data[2];
            let testingOutputData = iris_data[3];

            // Reshape input data to mach the input required by RNN models
            trainingData = trainingData.reshape([130, 1, 4]);
            testingData = testingData.reshape([14, 1, 4]);

            // Creating Model
            rnn_model = tf.sequential();

            rnn_model.add(tf.layers.simpleRNN(
                {
                    inputShape: [1, 4], // [number of time steps, number of features]
                    activation: 'sigmoid',
                    units: 5,
                    return_sequences: true
                }
            ));

            rnn_model.add(tf.layers.dense(
                {
                    inputShape: [130, 4],
                    units: 3,
                    activation: 'softmax'
                }
            ));

            console.log(rnn_model.summary());
            // compiling model
            rnn_model.compile({
                loss: "categoricalCrossentropy",
                metrics: ["accuracy"],
                optimizer: tf.train.adam(.06)
            })

            rnn_model.fit(trainingData, outputData, {
                epochs: epochs, shuffle: false, callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        //console.log("EPOCH END");
                        console.log("Loss: ", logs.loss);
                        loss = logs.loss;
                        console.log("Training Accuracy: ", logs.acc);
                        training_accuracy = logs.acc;

                    }
                }
            }).then(
                (history) => {
                    const evaluation = rnn_model.evaluate(testingData, testingOutputData)
                    // // Loss
                    // console.log("Testing Loss: ");
                    // console.log(evaluation[0].print());

                    // Testing Accuracy
                    console.log("Testing Accuracy: ");
                    evaluation[1].array().then(array => {
                        accuracy = array;
                        console.log(this.accuracy);
                    });

                    console.log("DONE");
                });
        }

    }

    //_____________________CNN train_____________________________________
    CNN_train(args) {

        // Reset values
        this.reset_values();

        if (args.Data == "MNIST") {
            // load MNIST data
            let data;
            async function load_data() {
                console.log('Loading MNIST data...');
                data = new MnistData();
                await data.load();
            }

            // train model
            load_data().then(() => {
                console.log('MNIST data is done loading');
                train_model().then(() => {
                    console.log('CNN model is done training...');

                    // const testExamples = 100;
                    // const examples = data.getTestData(testExamples);
                    // // The tf.tidy callback runs synchronously.
                    // tf.tidy(() => {
                    //     if (cnn_model != null) {
                    //         console.log("predicting");
                    //         console.log(examples.xs);
                    //         const output = cnn_model.predict(examples.xs);
                    //         const axis = 1;
                    //         const labels = Array.from(examples.labels.argMax(axis).dataSync());
                    //         const predictions = Array.from(output.argMax(axis).dataSync());

                    //         console.log(predictions);
                    //         //ui.showTestResults(examples, predictions, labels);
                    //     }
                    //});
                });
            });
        }

        async function train_model() {

            // ----------------- Create Model ------------------------
            console.log('creating CNN model...');

            cnn_model = tf.sequential();

            cnn_model.add(tf.layers.conv2d({
                inputShape: [IMAGE_H, IMAGE_W, 1],
                kernelSize: 3,
                filters: 16,
                activation: 'relu'
            }));

            // MaxPooling layer. This acts as a sort of downsampling using max values
            cnn_model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

            // Our third layer is another convolution, this time with 32 filters.
            cnn_model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }));
            // Max pooling again.
            cnn_model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

            // Add another conv2d layer.
            cnn_model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }));

            // flatten the output from the 2D filters into a 1D vector to prepare
            cnn_model.add(tf.layers.flatten({}));
            cnn_model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
            // Our last layer has 10 output units, one for each output class
            cnn_model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
            cnn_model.summary();
            // ----------------- End Create Model ------------------------

            // ----------------- Train Model ------------------------
            console.log('training CNN model......');

            // An optimizer is an iterative method for minimizing an loss function.
            const optimizer = 'rmsprop';

            cnn_model.compile({
                optimizer,
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy'],
            });

            // Batch size is hyperparameter that defines the number of examples to batch.
            // small value will update weights using few examples and will not generalize well.
            // Larger batch sizes require more memory resources.
            let batchSize = BATCH_SIZE_DEFAULT;
            if (args.BS) {
                batchSize = Number(args.BS);
            }

            // 15% of the training data for validation, to monitor overfitting
            const validationSplit = 0.15;

            // Get number of training epochs from the UI.
            //const trainEpochs = ui.getTrainEpochs();
            console.log(epochs)
            //const trainEpochs = epochs//10;

            // keep a buffer of loss and accuracy values over time.
            let trainBatchCount = 0;

            const trainData = data.getTrainData();
            const testData = data.getTestData();

            const totalNumBatches =
                Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
                epochs;

            let valAcc;
            await cnn_model.fit(trainData.xs, trainData.labels, {
                batchSize,
                validationSplit,
                epochs: epochs,
                callbacks: {
                    onBatchEnd: async (batch, logs) => {
                        trainBatchCount++;
                        /* ui.logStatus(
                        `Training... (` +
                        `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
                        ` complete). To stop training, refresh or close page.`);
                        ui.plotLoss(trainBatchCount, logs.loss, 'train');
                        ui.plotAccuracy(trainBatchCount, logs.acc, 'train');*/
                        if (batch % 10 === 0) {
                            console.log(batch);
                            console.log(logs);
                            //onIteration('onBatchEnd', batch, logs);
                        }
                        console.log("batch end");
                        training_accuracy = logs.acc;
                        loss = logs.loss;
                        await tf.nextFrame();
                    },
                    onEpochEnd: async (epoch, logs) => {
                        valAcc = logs.val_acc;
                        //ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
                        //ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
                        //console.log(epoch);
                        //console.log(logs);
                        // if (onIteration) {
                        //   onIteration('onEpochEnd', epoch, logs);
                        // }
                        console.log("epoch end");
                        await tf.nextFrame();
                    }
                }
            });

            //console.log(testData.xs);
            //console.log(testData.labels);
            const testResult = cnn_model.evaluate(testData.xs, testData.labels);
            const testAcc = testResult[1].dataSync()[0];
            //const testAccPercent = testAcc * 100;
            const finalValAccPercent = valAcc * 100;
            console.log(finalValAccPercent);
            accuracy = testAcc;
            // ----------------- End Train Model ------------------------
        }

    }

    //_____________________CNN predict_____________________________________

    predictMNIST(args) {

        // capture a video frame and add it to img element
        this.canvas.height = this.video.height;
        this.canvas.width = this.video.width;
        var ctx = this.canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0, this.canvas.height, this.canvas.width);
        var img = new Image();
        img.src = this.canvas.toDataURL();

        // var d=this.canvas.toDataURL("image/png");
        // var w=window.open('about:blank','image from canvas');
        // w.document.write("<img src='"+d+"' alt='from canvas'/>");

        // convert canvas to tensor
        let tensor = tf.browser.fromPixels(this.canvas, 1).resizeBilinear([28, 28]);
        const eTensor = tensor.expandDims(0);
        //console.log("eTensor shape");
        //console.log(eTensor);

        // // make predictions on the preprocessed image tensor
        let output = cnn_model.predict(eTensor);
        const axis = 1;
        //const labels = Array.from(examples.labels.argMax(axis).dataSync());
        const MNIST_prediction = Array.from(output.argMax(axis).dataSync());
        console.log(MNIST_prediction);
        prediction = MNIST_prediction;

    }

    // Pre-trained model
    detectObj(args) {
        cocoSsd.load().then(model => {
            // detect objects in the Video.
            model.detect(this.video).then(predictions => {
                console.log('Predictions: ', predictions);
                var i;
                for (i = 0; i < predictions.length; i++) {
                    var index = i + 1;

                    if (index < 4) {
                        this.objectClasses[i].value = predictions[i].class;
                        this.objectScores[i].value = predictions[i].score;
                        let pos = this.convertPositions(predictions[i].bbox[0], predictions[i].bbox[1])
                        this.objectXPos[i].value = pos[0];
                        this.objectYPos[i].value = pos[1];
                        console.log(this.objectYPos[i]);
                        console.log(this.objectXPos[i]);
                        console.log(predictions[i].class + " x:" + predictions[i].bbox[0] + "y:" + predictions[i].bbox[1]);
                        console.log(pos[0]);
                    }
                    else {
                        break;
                    }
                }

            });
        });
    }

    objectClass(args) {
        let index = this.objectClasses.findIndex(x => x.text === args.OBNO);
        return this.objectClasses[index].value;
    }

    objectScore(args) {
        let index = this.objectScores.findIndex(x => x.text === args.OBSCSCORES);
        return this.objectScores[index].value;
    }

    objectxPos(args) {
        let index = this.objectXPos.findIndex(x => x.text === args.OBX);
        return this.objectXPos[index].value;
    }

    objectyPos(args) {
        let index = this.objectYPos.findIndex(x => x.text === args.OBY);
        return this.objectYPos[index].value;
    }

    videoToggle(args) {

        let state = args.VIDEO_STATE;
        if (state === 'off') {
            this.runtime.ioDevices.video.disableVideo();
        } else {
            let media = navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false,
            });

            media.then((stream) => {
                this.video.srcObject = stream;
            });
            this.runtime.ioDevices.video.enableVideo();
            this.runtime.ioDevices.video.mirror = state === "on";
        }
    }

    getVideoMenu() {
        return [
            {
                text: Message.off[this.locale],
                value: 'off'
            },
            {
                text: Message.on[this.locale],
                value: 'on'
            },
            {
                text: Message.video_on_flipped[this.locale],
                value: 'on-flipped'
            }
        ]
    }

    getCLassMenu() {
        return [
            {
                text: '1',
                value: '1'
            },
            {
                text: '2',
                value: '2'
            },
            {
                text: '3',
                value: '3'
            }
        ];
    }

    getScoreMenu() {
        return [
            {
                text: '1',
                value: '1'
            },
            {
                text: '2',
                value: '2'
            },
            {
                text: '3',
                value: '3'
            }
        ];
    }

    getXPosMenu() {
        return [
            {
                text: '1',
                value: '1'
            },
            {
                text: '2',
                value: '2'
            },
            {
                text: '3',
                value: '3'
            }
        ];
    }

    getYPosMenu() {
        return [
            {
                text: '1',
                value: '1'
            },
            {
                text: '2',
                value: '2'
            },
            {
                text: '3',
                value: '3'
            }
        ];
    }

    setLocale() {
        let locale = formatMessage.setup().locale;
        if (AvailableLocales.includes(locale)) {
            return locale;
        } else {
            return 'en';
        }
    }

    convertPositions(x, y) {
        let w2 = this.video.width / 2
        let h2 = this.video.height / 2
        if (x == w2 && y == h2) {
            return [0, 0]
        } else if (x < w2 && y < h2) {
            let newx = w2 - x
            let newy = h2 - y
            return [newx, newy]
        } else if (x > w2 && y > h2) {
            let newx = x - w2
            let newy = y - h2
            newx = w2 - x
            newy = h2 - y
            if (newx > 0) {
                newx = newx * (-1)
            }
            if (newy > 0) {
                newy = newy * (-1)
            }
            return [newx, newy]
        } else if (x > w2 && y < h2) {
            let newx = x - w2
            let newy = h2 - y
            newx = w2 - x
            if (newx > 0) {
                newx = newx * (-1)
            }
            return [newx, newy]
        } else if (x < w2 && y > h2) {
            let newx = w2 - x
            let newy = y - h2
            newy = h2 - y
            if (newy > 0) {
                newy = newy * (-1)
            }
            return [newx, newy]
        } else if (x == w2 && y < h2) {
            let newy = h2 - y
            return [0, newy]
        } else if (x == w2 && y > h2) {
            let newy = y - h2
            newy = h2 - y
            if (newy > 0) {
                newy = newy * (-1)
            }
            return [0, newy]
        } else if (x > w2 && y == h2) {
            let newx = x - w2
            newx = w2 - x
            if (newx > 0) {
                newx = newx * (-1)
            }
            return [newx, 0]
        } else if (x < w2 && y == h2) {
            let newx = w2 - x
            return [newx, 0]
        }
    }

} // END Class

module.exports = Scratch3DeepScratch;
