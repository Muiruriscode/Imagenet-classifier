# Pre-trained Imagenet classifier
This is a project using the pre-trained moblienet classifier from tensorflow hub. It is is already trained with to detect 1000 objects. You can add capabilities to the model to detect your own custom objects. This is  graph model therefore optimized for performance.

## About
The model detects two oblects based on which button has been cklicked. If the first button is pressed it will gather data for an object and objects will be classified as class 1. If class two button is pressed the model wil detect objects which will be classified as class 2. 
More buttons bcan be added to classify more objects since the process is dynamic.

!["class 1"](./assets/class%201.png)

## Check video support and load video
```js
function hasGetUserMedia(){
    //check for mediaDevice support and getUser media support in the browser
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
}

function enableCam(){
    if(hasGetUserMedia()){
        const constraints = {
            video: true,
            width: 640,
            height: 480,
        }

    //allow streaming through the webcam using the browser
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream){
            VIDEO.srcObject = stream
            VIDEO.addEventListener('loadeddata', function(){
                videoPlaying = true
                ENABLE_CAM_BUTTON.classList.add('removed')
            })
        })
    }else{
        console.warn('getUserMedia() is not supported your browser')
    }
}

```

## Gather data 
Gather data accoring to the button supported. Then change the state of gather data to eneble you to gather video data through the web cam.
```js
function gaterDataForClass() {
    let classNumber = parseInt(this.getAttribute('data-1hot'));
    gatherDataState = (gatherDataState === STOP_GATHER_DATA) ? classNumber : STOP_GATHER_DATA;
    dataGatherLoop();
  }

function calculateFeaturesOnCurrentFrame() {
    return tf.tidy(function() {
      // Grab pixels from current VIDEO frame.
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
      // Resize video frame tensor to be 224 x 224 pixels which is needed by MobileNet for input.
      let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor, 
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
      );
  
    //normalize the input
      let normalizedTensorFrame = resizedTensorFrame.div(255);
  
    //expand the dimentions of normalizedTesnsorFrame to allow fro batch size
    //squeeze normalizedTensorFrame to return it from the function
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });
  }

function dataGatherLoop() {
    // Only gather data if webcam is on and a relevent button is pressed.
    if (videoPlaying && gatherDataState !== STOP_GATHER_DATA) {
      // Ensure tensors are cleaned up.
      let imageFeatures = calculateFeaturesOnCurrentFrame();
  
      trainingDataInputs.push(imageFeatures);
      trainingDataOutputs.push(gatherDataState);
      
      // Intialize array index element if currently undefined.
      if (examplesCount[gatherDataState] === undefined) {
        examplesCount[gatherDataState] = 0;
      }
      // Increment counts of examples for user interface to show.
      examplesCount[gatherDataState]++;
  
      STATUS.innerText = '';
      for (let n = 0; n < CLASS_NAMES.length; n++) {
        STATUS.innerText += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '. ';
      }
  
      window.requestAnimationFrame(dataGatherLoop);
    }
  }

```

## Train and predict
This process involves training the model and then prdicting the value of objects detected in video frame. 
```js
async function trainAndPredict() {
    predict = false;
    // shuffle input and output concurrently relative to their position
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
    
    // convert traingData outputs into a tensor1d in the int 32 format
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    //oneHot encode the output
    let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    //convert training data inputs into a tensor2d using tf.stsck
    let inputsAsTensor = tf.stack(trainingDataInputs);
    
    //train the data 
    let results = await model.fit(inputsAsTensor, oneHotOutputs, {
      shuffle: true,
      batchSize: 5,
      epochs: 10,
      callbacks: {onEpochEnd: logProgress}
    });
    
    //dispose tensors from memory
    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    
    predict = true;
    predictLoop();
  }

  /**
   *  Make live predictions from webcam once trained.
   **/
  function predictLoop() {
    if (predict) {
      tf.tidy(function() {
        let imageFeatures = calculateFeaturesOnCurrentFrame();
        let prediction = model.predict(imageFeatures.expandDims()).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let predictionArray = prediction.arraySync();
        STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
      });
  
      window.requestAnimationFrame(predictLoop);
    }
  }
  
  ```
