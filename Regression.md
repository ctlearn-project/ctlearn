### Introduction and Opening Blurb
Hello everyone! My name is Evan and I am in my final year of study at Nanyang Technical University, Singapore where I am majoring in Physics. My degree program has provided me a substantial background in mathematics. It has also covered topics in applied particle physics such as Hadronic/Electro-magnetic cascades and Cherenkov Radiation itself which aids in understanding Cherenkov Telescope operation, as well as its role in both in high-energy physics – using the universe as a particle collider, and in answering unsolved questions in astrophysics.

I have built the CTLearn enviroment as described in the documentation and am still in the midst of familarizing myself with the code base. A summary of my project proposal is as follows.

### Project Proposal Outline
* A `continuous_load` function based on the work done in Dl1DataHandler will be used to construct a data pipeline for our Multi-Task Learning Model.
* The model will be subdivided into two main components: `feature extractor` and `multi-task estimator`
* The first iteration of `feature extractor` will work with single gamma-ray images.
* The  `multi-task estimator` will consist of multiple regression models, each constituting a single task that feeds from the final layer of `feature extractor`

The model for particle energy estimation utilising a single gamma ray image will be the main focus of development. Should time permit, this model will be extended to utilise multiple gamma-ray images corresponding to the same event but from different orientations. This model can also be further extended to predict particle arrival direction if there is sufficient time.
