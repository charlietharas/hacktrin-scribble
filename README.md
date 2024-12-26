Lets you write in the air with your pointer finger, and uses a simple neural net trained on EMNIST to tell you which letters you wrote!

Thanks to [Jason Wu](https://github.com/abstractlegwear) & Maggie Kwan for their help with the initial version of this back in 2021!

## Branches
- `main` has new Python implementation
- `old` has archived original submission for HackTrin in 2021
- `js` has Python implementation that saves model to web version, as well as web implementation of GUI

## Installation
`pip install opencv-python mediapipe tensorflow keras emnist`

If model weights not detected, program will attempt to generate model weights file in working directory. At time of publishing, `emnist` direct downloads are broken; follow error mesage instructions to get it to load the data.

If it's not working the way you expect/want, the first place to start is tweaking default params in function headers.

## Usage
Once the model is trained, it can be reloaded from the weights file.

Running will open a camera window. Simply:
- Use your index finger to draw
- Raise your middle finger by your index finger to finish the current letter; model predictions are printed to stdout
- Raise your middle and ring fingers to move your hand freely without clearing the letter
