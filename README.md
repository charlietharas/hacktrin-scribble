# hacktrin-scribble
Winner for Trinity High School's Hacktrin VIII (2021) Best Overall award. Contains incomplete prototype code, not the final submission.

Githubs:
- Jason Wu [@abstractlegwear](https://github.com/abstractlegwear).

Submission video [here](https://youtu.be/qcaZ_k0h4SU).

This code contains the two unconnected components of our project. The first component, in the file `EdgeDetection.py`, utilizes the OpenCV library to detect fingers from a user's camera after a quick setup process and then tracks the user's tracing of their finger through the air into letters. Every time the user wishes to enter a new letter into the tracker (being done drawing the old one) the user may forcefully blink, which will clear the board and process the previous trace. 

The second compnent is a letter classification model trained on the extended-MNIST "EMNIST" dataset. This model will parse 28 by 28 resizes of traced lines against its training on lower and uppercase 28 by 28 letters and numbers, and then output the result.

This repository does not feature the two code sections linked with each other. They currently operate independently. This code is also not optimized for independent deployment but rather as a source code storage. Big thanks to the helpful judges & staff at Hacktrin for letting us compete and setting up a wonderful event. Check them out at [their website](http://www.hacktrin.com/). Feel free to contact me or my partners via outreach at charlie@charliemax.dev.
