This is my implementation of a resnet using pytorch. This resnet is based on option (A) from the original resnet paper with identity connections and 0 padding between layers of different sizes. Run the CIFA10_ResNet.py file to run the analysis. You can configure the depth by setting the n parameter in the script to get a network depth of 2+6n. 

There is also a pretrained model available for analysis. In the testModel folder there is a checkImage.py which uses my n=5 resnet to analyze an input image. This code was trained on the CIFAR-10 dataset so the network should be able to recognize any image from the classes used in that dataset. It can be run as "python checkImage.py Images/<image_name>"

ResNet.py is a more generalized form of the resnet code, that is currently configured to create a resnet-50. This code uses the blocks of 3 convolutional layers most typically associated with deeper resnets, and uses projection shortcuts for shortcuts across layers of different sizes.

The results section contains my results for n = {3,5,7,9,18}, with iterations vs accuracy and saved models. 