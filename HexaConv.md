### Introduction and Opening Blurb
Hello everyone! My name is Evan and I am in my final year of study at Nanyang Technical University, Singapore where I am majoring in Physics. My degree program has provided me a substantial background in mathematics. It has also covered topics in applied particle physics such as Hadronic/Electro-magnetic cascades and Cherenkov Radiation itself which aids in understanding Cherenkov Telescope operation, as well as its role in both in high-energy physics – using the universe as a particle collider, and in answering unsolved questions in astrophysics.

 I have built the CTLearn enviroment as described in the documentation and am still in the midst of familarizing myself with the code base. A summary of my project proposal is as follows.

 ### Project Proposal Outline
In order to implement Hexagonal Convolution, hexagonal equivalents of the following `tf.layers` functions have to be written:
* `conv2d` -> `hexaconv2d`
* `max_pooling2d` -> `hex_max_pooling2d`
* `average_pooling2d` -> `hex_average_pooling2d`
We will also define a new `hexaconv_block` as an analogue to `conv_block`.

Additionally, if time permits, we will also implement `hexaconv3d` and `hex_max_pooling3d` as analogues to `conv3d` and `max_pooling3d`.

These functions will be written in Tensorflow as part of CTLearn and the goal is to have these functions have an identical API to that of `tf.layers`.
