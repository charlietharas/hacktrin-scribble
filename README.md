Lets you write in the air with your pointer finger, and uses a simple neural net trained on EMNIST to tell you which letters you wrote!

Thanks to [Jason Wu](https://github.com/abstractlegwear) & Maggie Kwan for their help with the initial version of this back in 2021!

## Installation
`pip install tensorflowjs tensorflow==2.15.0 tensorflow-decision-forests==1.8.1`

`pip install -I opencv-python mediapipe emnist`

If model weights not detected, program will attempt to generate model weights file in working directory. At time of publishing, `emnist` direct downloads are broken; follow error mesage instructions to get it to load the data.

Program outputs a converted tensorflowjs model into the `modelfile` directory after training; use this for a web implementation.

If it's not working the way you expect/want, the first place to start is tweaking default params in function headers.

## Usage
Once the model is trained, it can be reloaded from the weights file.

Running will open a camera window. Simply:
- Use your index finger to draw
- Raise your middle finger by your index finger to finish the current letter; model predictions are printed to stdout
- Raise your middle and ring fingers to move your hand freely without clearing the letter
