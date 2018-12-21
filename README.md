# disentangled-caption-generator
In this repository is the code for a solution of the captioning problem. The goal was to find a method of disentangling the two tasks of the decoder part of a caption generator. The two tasks are, generating grammatically correct sentences and generating a good description of the image. 

In this repository is the code for the baseline and the three sub-models of the disentanglement model.

Before one can run this code, the datasets must be downloaded and added under the data folder. The labels of the data require to be in a special format for the evaluation code. For the Flickr datasets a script is created that can convert this. The conversions for Flickr8k and the Flickr30k are already in the repository. 

### directories
In the main directory are all the code files for running the program. 
There are several subdirectories where files are stored: 
 - data: here are all the files for different datasets
 - pickles: all the pickles used for running the program, such as the vocabulary
 - caption eval: the code for running evaluations
 - output: here will the output be stored after running
 - pycocotools: some helper file for coco evaluation
 
### References
In this project code has been used from the following sources.
 - https://github.com/tylin/coco-caption
 - https://github.com/MaximumEntropy/Seq2Seq-PyTorch

Note that it has been adapted for our needs. One does not require to clone these before using this project.  
