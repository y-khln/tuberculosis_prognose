const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const path = require('path');

//определяем папки с изображениями для обучения и тестирования
const TRAIN_IMAGES_DIR = './data/train';
const TEST_IMAGES_DIR = './data/test';

//чтение картинок-снимков
function loadImages(dataDir) {
  const images = [];
  const labels = [];
  
  var files = fs.readdirSync(dataDir);
  for (let i = 0; i < files.length; i++) { 
    if (!files[i].toLocaleLowerCase().endsWith(".png")) {
      continue;
    }

    var filePath = path.join(dataDir, files[i]);
    
    var buffer = fs.readFileSync(filePath);
    //преобразовать в тензор, задав один размер изображениям
    var imageTensor = tf.node.decodeImage(buffer)
      .resizeNearestNeighbor([96,96])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();
    images.push(imageTensor);

    var hasTuberculosis = files[i].toLocaleLowerCase().endsWith("_1.png");
    labels.push(hasTuberculosis ? 1 : 0);
  }

  return [images, labels];
}

//класс для работы с датасетом
class TuberculosisDataset {
  constructor() {
    this.trainData = [];
    this.testData = [];
  }

  //загрузка изображений
  loadData() {
    console.log('Loading images...');
    this.trainData = loadImages(TRAIN_IMAGES_DIR);
    this.testData = loadImages(TEST_IMAGES_DIR);
    console.log('Images loaded successfully.')
  }
  //геттер для снимков из обучающего множества
  getTrainData() {
    return {
      images: tf.concat(this.trainData[0]),
      labels: tf.oneHot(tf.tensor1d(this.trainData[1], 'int32'), 2).toFloat()
    }
  }
  //геттер для тестовых снимков
  getTestData() {
    return {
      images: tf.concat(this.testData[0]),
      labels: tf.oneHot(tf.tensor1d(this.testData[1], 'int32'), 2).toFloat()
    }
  }
}

module.exports = new TuberculosisDataset();
