const tf = require('@tensorflow/tfjs');
const iris = require('./training.json');
const irisTesting = require('./testing.json');

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Cast = require('../../util/cast');
const log = require('../../util/log');

const EPOCHS_DEFAULT = 10;
const LAYERS_DEFAULT = 2;

class Scratch3DeepScratch {

  constructor (runtime) {
    this.runtime = runtime;
    this.epochs = EPOCHS_DEFAULT;
    this.layers = LAYERS_DEFAULT;
    this.accuracy = "";
    this.training_accuracy = "";
    this.loss = "";
    this.model = null;
    this.prediction = "";
  }

  getInfo () {
    return {
      id: 'deepscratch',
      color1: '#1ebdad',
      color2: '#0aab9b',
      name: 'Deep Scratch',
      blocks: [
        {
          opcode: 'train',
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
        dataMenu: ['iris', 'ImgNet'],
        modelsMenu: ['Dense', 'RNN', 'CNN']
      }
    };
  }

  getAccuracy(){
    //console.log(this.accuracy);
    return this.accuracy;
  }

  getTrainingAccuracy(){
    return this.training_accuracy;
  }

  getLoss(){
    //console.log(this.loss);
    return this.loss;
  }

  getPrediction() {
    return this.prediction;
  }
  setLayers (args) {
    this.layers = args.Number;
  }

  setEpochs (args) {
    this.epochs = args.Number;
  }

  train (args) {

    this.accuracy = "";
    this.training_accuracy = "";
    this.loss = "";
    this.prediction = "";

    if (args.Data == "iris") {

      // Mapping the trainingdata
      const trainingData = tf.tensor2d(iris.map(item=> [
        item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
      ]
    ),[130,4])

    // Mapping the testing data
    const testingData = tf.tensor2d(irisTesting.map(item=> [
      item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
    ]
  ),[14,4])

  // Mapping the output data
  const outputData = tf.tensor2d(iris.map(item => [
    item.species === 'setosa' ? 1 : 0,
    item.species === 'virginica' ? 1 : 0,
    item.species === 'versicolor' ? 1 : 0

  ]), [130,3])

  // Mapping the output data
  const testingOutputData = tf.tensor2d(irisTesting.map(item => [
    item.species === 'setosa' ? 1 : 0,
    item.species === 'virginica' ? 1 : 0,
    item.species === 'versicolor' ? 1 : 0

  ]), [14,3])

  // Creating Model
  this.model = tf.sequential();

  if (args.Model == "Dense") {

    this.model.add(tf.layers.dense(
      {   inputShape: 4,
        activation: 'sigmoid',
        units: 5
      }
    ));

    for(let i = 1; i <= this.layers; i++) {
      this.model.add(tf.layers.dense(
        {   inputShape: 5,
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

    console.log(this.model.summary());
    // compiling model
    this.model.compile({
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
      optimizer: tf.train.adam(.06)
    })

    this.model.fit(trainingData, outputData, {epochs: this.epochs, shuffle: false, callbacks: {
      onEpochEnd: (epoch, logs) => {
        //console.log("EPOCH END");
        console.log("Loss: ", logs.loss);
        this.loss = logs.loss;
        console.log("Training Accuracy: ", logs.acc);
        this.training_accuracy = logs.acc;

      }
    }}).then(
      (history)=>{
        const evaluation = this.model.evaluate(testingData, testingOutputData)
        // // Loss
        // console.log("Testing Loss: ");
        // console.log(evaluation[0].print());

        // Testing Accuracy
        console.log("Testing Accuracy: ");
        evaluation[1].array().then(array => {
          this.accuracy = array;
          console.log(this.accuracy);
        });


        //console.log(model.predict(testingData).print());
        //
        // console.log("yTrue: ");
        // console.log(testingOutputData.print());
        // console.log("yPred: ");
        // yPred.print();
        // const recall = tf.metrics.precision(testingOutputData, yPred);
        // console.log("RECALL: ");
        // console.log(recall.print());

        console.log("DONE");
      }
    );

  }

}

this.epochs = EPOCHS_DEFAULT;
this.layers = LAYERS_DEFAULT;

}

predictIris(args) {

  this.prediction = "";

  if (this.model != null) {

    const newData = tf.tensor2d([[Number(args.sepal_length), Number(args.sepal_width), Number(args.petal_length), Number(args.petal_width)]]);

    const prediction = this.model.predict(newData).array().then(array => {
      array = array[0];
      let max = array[0]
      if (array[1] > max) {
        max = array[1];
      }
      if (array[2] > max) {
        max = array[2];
      }
      // console.log(array[0]);
      // console.log(array[1]);
      // console.log(array[2]);
      // console.log(max);

      if (max == array[0]) {
        this.prediction = "setosa";
      } else if (max == array[1]) {
        this.prediction = "virginica";
      } else {
        this.prediction = "versicolor";
      }

      //console.log(this.prediction);
    });

  }
}
}

module.exports = Scratch3DeepScratch;
