# Deep Saliency Map Generators for Multispectral Video Classification

This is the official repository for the ICPR 2022 Paper: Deep Saliency Map Generators for Multispectral Video Classification.

## Abstract

Despite their black box nature, deep neural networks have been successfully 
used in practical applications lately. In areas where the results of these applications can 
lead to safety hazards or decisions of ethical relevance, the application 
provider is accountable for the resulting decisions and should therefore 
be able to explain, how, and why a specific decision was made.
For image processing networks, saliency map generators are a possible solution. A 
saliency map gives a visual hint on what is of special importance for the network's
decision, can reveal possible dataset biases and give a more profound insight in the 
decision process of the black box.

This paper investigates how 2D saliency map generators need to be adapted for 3D input data,
and additionally, how the methods behave when applied not only to ordinary video input but 
rather multispectral 3D input data.
This is exemplarily shown on 3D video input data in human action recognition in the infrared and visual
spectrum and evaluated by using the insertion and deletion metrics. The dataset used in this work is 
the Multispectral Action Dataset, where each scene is available in the long-wave infrared as well as 
the visual spectrum. To be able to draw a more general conclusion, the two investigated networks,
3D-ResNet 18 and Persistent Appearance Network (PAN), follow a different mindset.

It could be shown, that the saliency methods can also be applied to 3D input data with remarkable results.
The results show that a combined training with both, infrared and RGB 3D input data, lead to more
focused saliency maps and outperform a training with only RGB or infrared data.


## Setup

Clone this repository
> git clone https://github.com/JensBayer/ICPR2022.git icpr2022-video-classification-saliency

Initialize the git-submodules and patch it
> cd icpr2022-video-classification-saliency

> git submodules init

> git submodule update

> cd PAN-PyTorch && git apply ../PAN-patch.diff

Create and activate a new conda environment
> conda env create -n icpr2022-video-classification-saliency --file=environment.txt

> conda activate icpr2022-video-classification-saliency

and open one of the jupyter notebooks (`C3D_evaluation.ipynb` or `PAN_evaluation.ipynb`).


## Dataset
The dataset can freely be requested here: 
> https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/vca.html

## Weights
The pre-trained weights can be downloaded here: 
> https://github.com/JensBayer/ICPR2022/releases/tag/v1
