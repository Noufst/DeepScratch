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

let cnn_model = null;

class Scratch3DeepScratch {

    constructor(runtime) {
        this.runtime = runtime;
        this.epochs = EPOCHS_DEFAULT;
        this.layers = LAYERS_DEFAULT;
        this.accuracy = "";
        this.training_accuracy = "";
        this.loss = "";
        this.dense_model = null;
        this.model_type = "";
        this.prediction = "";
        this.video = document.createElement("video");
        this.video.width = 408;
        this.video.height = 306;
        this.video.autoplay = true;
        this.video.style.display = "none";
        this.canvas = document.createElement('canvas');

    }

    getInfo() {

        return {
            id: 'deepscratch',
            color1: '#1ebdad',
            color2: '#0aab9b',
            name: 'Deep Scratch',
            blocks: [
                {
                    opcode: 'dense_train',
                    blockType: BlockType.COMMAND,
                    text: 'train [Data] with model [Model]',
                    arguments: {
                        Data: {
                            type: ArgumentType.STRING,
                            menu: 'dataMenu'
                        },
                        Model: {
                            type: ArgumentType.STRING,
                            menu: 'modelsMenu'
                        }
                    }
                },
                {
                    opcode: 'CNN_predict',
                    blockType: BlockType.COMMAND,
                    text: 'predict MNIST with CNN',
                },
                {
                    opcode: 'launch_camera',
                    blockType: BlockType.COMMAND,
                    text: 'lanuch camera',
                },
                {
                    opcode: 'CNN_train',
                    blockType: BlockType.COMMAND,
                    text: 'train MNIST with CNN',
                },
                {
                    opcode: 'setEpochs',
                    blockType: BlockType.COMMAND,
                    text: 'set number of epochs [Number]',
                    arguments: {
                        Number: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 10
                        }
                    }
                },
                {
                    opcode: 'setLayers',
                    blockType: BlockType.COMMAND,
                    text: 'set number of hidden layers [Number]',
                    arguments: {
                        Number: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 2
                        }
                    }
                },
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
                }
            ],
            menus: {
                dataMenu: ['iris', 'ImgNet', 'MNIST'],
                modelsMenu: ['Dense', 'RNN', 'CNN']
            }
        };
    } // End getinfo

    getAccuracy() {
        //console.log(this.accuracy);
        return this.accuracy;
    }

    getTrainingAccuracy() {
        return this.training_accuracy;
    }

    getLoss() {
        //console.log(this.loss);
        return this.loss;
    }

    getPrediction() {
        return this.prediction;
    }

    setLayers(args) {
        this.layers = args.Number;
    }

    setEpochs(args) {
        this.epochs = args.Number;
    }

    dense_train(args) {

        this.accuracy = "";
        this.training_accuracy = "";
        this.loss = "";
        this.prediction = "";
        this.model_type = args.Model;

        // ___________________________IRIS data ____________________________
        if (args.Data == "iris") {

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

        } // End iris data

        // Creating Model
        this.dense_model = tf.sequential();


        this.dense_model.add(tf.layers.dense(
            {
                inputShape: 4,
                activation: 'sigmoid',
                units: 5
            }
        ));

        for (let i = 1; i <= this.layers; i++) {
            this.model.add(tf.layers.dense(
                {
                    inputShape: 5,
                    activation: 'sigmoid',
                    units: 5
                }
            ));
        }

        this.model.add(tf.layers.dense(
            {
                inputShape: 5,
                units: 3,
                activation: 'softmax'
            }
        ));


        // ___________________________RNN Model ____________________________

        // else if (this.model_type == "RNN") {
        //     // Reshape input data to mach the input required by RNN models
        //     trainingData = trainingData.reshape([130, 1, 4]);
        //     testingData = testingData.reshape([14, 1, 4]);

        //     this.model.add(tf.layers.simpleRNN(
        //         {
        //             inputShape: [1, 4], // [number of time steps, number of features]
        //             activation: 'sigmoid',
        //             units: 5,
        //             return_sequences: true
        //         }
        //     ));

        //     this.model.add(tf.layers.dense(
        //         {
        //             inputShape: [130, 4],
        //             units: 3,
        //             activation: 'softmax'
        //         }
        //     ));
        // }
        // 
        this.epochs = EPOCHS_DEFAULT;
        this.layers = LAYERS_DEFAULT;

    }

    //_____________________CNN train_____________________________________
    CNN_train(args) {

        this.accuracy = "";
        this.training_accuracy = "";
        this.loss = "";
        this.prediction = "";

        let data;

        // load data
        async function load_data() {
            console.log('Loading data...');
            data = new MnistData();
            await data.load();
        }

        // train model
        load_data().then(() => {
            console.log('data is done loading');
            train_model().then(() => {
                console.log('model is done training...');
                const testExamples = 100;
                const examples = data.getTestData(testExamples);

                // The tf.tidy callback runs synchronously.
                tf.tidy(() => {
                    if (cnn_model != null) {
                        console.log("predicting");
                        console.log(examples.xs);
                        const output = cnn_model.predict(examples.xs);
                        const axis = 1;
                        const labels = Array.from(examples.labels.argMax(axis).dataSync());
                        const predictions = Array.from(output.argMax(axis).dataSync());

                        console.log(predictions);
                        //ui.showTestResults(examples, predictions, labels);
                    }
                });
            });
        });

        async function train_model() {

            // const trainData = data.getTrainData();
            // console.log("*********");
            // console.log(trainData.labels);
            // console.log(trainData.xs);
            // console.log("*********");

            console.log('creating model...');

            // ----------------- Create Model ------------------------
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
            console.log('model is training...');


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
            const batchSize = 320;

            // 15% of the training data for validation, to monitor overfitting
            const validationSplit = 0.15;

            // Get number of training epochs from the UI.
            //const trainEpochs = ui.getTrainEpochs();
            const trainEpochs = 10;

            // keep a buffer of loss and accuracy values over time.
            let trainBatchCount = 0;

            const trainData = data.getTrainData();
            const testData = data.getTestData();

            const totalNumBatches =
                Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
                trainEpochs;

            let valAcc;
            await cnn_model.fit(trainData.xs, trainData.labels, {
                batchSize,
                validationSplit,
                epochs: trainEpochs,
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

            console.log("fitting end");
            console.log(testData.xs);
            console.log(testData.labels);
            const testResult = cnn_model.evaluate(testData.xs, testData.labels);
            const testAccPercent = testResult[1].dataSync()[0] * 100;
            const finalValAccPercent = valAcc * 100;
            console.log('model is done training...');
            console.log(finalValAccPercent);
            // ----------------- End Train Model ------------------------
        }

    }

    //_____________________CNN predict_____________________________________
    launch_camera(args) {

        let media = navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
        });

        media.then((stream) => {
            this.video.srcObject = stream;
        });

        this.runtime.ioDevices.video.enableVideo();
    }

    CNN_predict(args) {


        this.canvas.height = this.video.height;
        this.canvas.width = this.video.width;
        var ctx = this.canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0, this.canvas.height, this.canvas.width);
        var img = new Image();
        img.src = this.canvas.toDataURL();

        // var d=this.canvas.toDataURL("image/png");
        // var w=window.open('about:blank','image from canvas');
        // w.document.write("<img src='"+d+"' alt='from canvas'/>");

        // preprocess canvas
        let tensor = tf.browser.fromPixels(this.canvas, 1).resizeBilinear([28, 28]);
        // .resizeNearestNeighbor([28, 28])
        // .mean(2)
        // .expandDims(2)
        // .expandDims()
        // .toFloat()
        // .div(255.0);
        console.log("tensor shape");
        console.log(tensor);
        const eTensor = tensor.expandDims(0);
        console.log("eTensor shape");
        console.log(eTensor);
        // // make predictions on the preprocessed image tensor
        let output = cnn_model.predict(eTensor);

        const axis = 1;
        //const labels = Array.from(examples.labels.argMax(axis).dataSync());
        const predictions = Array.from(output.argMax(axis).dataSync());
        console.log(predictions);

    }

    //____________________ Prediction _____________________________

    predictIris(args) {

        this.prediction = "";

        if (this.model != null) {

            let newData = tf.tensor2d([[Number(args.sepal_length), Number(args.sepal_width), Number(args.petal_length), Number(args.petal_width)]]);

            if (this.model_type == "RNN") {
                // Reshape input for RNN
                newData = newData.reshape([1, 1, 4]);
            }

            const prediction = this.model.predict(newData).array().then(array => {

                array = array[0];
                let max = array[0]
                if (array[1] > max) {
                    max = array[1];
                }
                if (array[2] > max) {
                    max = array[2];
                }

                if (max == array[0]) {
                    this.prediction = "setosa";
                } else if (max == array[1]) {
                    this.prediction = "virginica";
                } else {
                    this.prediction = "versicolor";
                }
            });

        }
    } // END predict iris


} // END Class

module.exports = Scratch3DeepScratch;
