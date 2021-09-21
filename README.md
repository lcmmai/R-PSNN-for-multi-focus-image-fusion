# R-PSNN-for-multi-focus-image-fusion

Code for the paper: Pseudo-Siamese residual atrous pyramid network for multi-focus image fusion
A simple and fast multi-focus image fusion method R-PSNN based on regression model.

# Abstract: 
Depth of field is one of the critical reasons to limit the richness of image information. Usually, in a scene with multiple targets, when the distance between each target and the lens is different, the clear scene image can be get within a certain distance range. This situation restricts the further image processing, such as semantic segmentation, object recognition and 3D reconstruction. Multi-focus image fusion uses two or more images focused on different targets to fuse scene information, which can solve this problem to a great extent. In general, two or more multi-focus images can cover almost all near/far targets. The fusion of more than two multi-focus images can be accomplished by cascading the fusion results of the previous two images and the next image to be processed many times. Therefore, the paper focus on the fusion of two multi-focus images. Inspired by this, new Pseudo-Siamese neural network with several residual atrous convolution pyramids with multi-level perception ability to perceive the multi-level features and consistency relations of multi-focus image pairs is proposed, and multi-layer residual blocks are used to fuse the extracted features. In this process, the residual of the groundtruth and the generated image will be learned. Finally, a fully focused image without blur will be generated. After several ablation experiments and comparison experiments with other methods, the results show that the performance of the method proposed in this paper is state-of the-art, and overall better than other methods, which are advanced.

# Citation
If you want to use this code in your research, please cite:
Jiang, L., Fan, H., Li, J., & Tu, C. (2021). Pseudo‐Siamese residual atrous pyramid network for multi‐focus image fusion. IET Image Processing. https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12326
````
@article{jiang2021pseudo,
  title={Pseudo-Siamese residual atrous pyramid network for multi-focus image fusion},
  author={Jiang, Limai and Fan, Hui and Li, Jinjiang and Tu, Changhe},
  journal={IET Image Processing},
  year={2021},
  publisher={Wiley Online Library}
}
````

# Requirements
The code is based on Pytorch, you can use an Anaconda environment and download the requirements.

# Experiments
![BNT{LR}JK6Z418QNY 55Y9R](https://user-images.githubusercontent.com/76153473/134174737-3fa5c806-b28f-4de2-980b-600f0477564a.png)
The overall network structure of R-PSNN.

![metric](https://user-images.githubusercontent.com/76153473/134173286-c26be936-7b65-4a25-a7a2-d82dda5b169c.png)
The objective metrics comparison of various methods in several fusion images, The colored values are the best experimental results.

We also provide the original results of ‘Comparison of objective metrics’, see 'metrics.xlsx' for detials.
For the citation of evaluation metrics, please refer to the relevant section(4.4.1) in the paper. Our paper is open source.

![9$J0G9IATKTCCHIJ44F@9WW](https://user-images.githubusercontent.com/76153473/134175024-a5d7a10a-0d35-489a-9240-c25fd4ed1b9e.png)
Performance and detail comparison of all methods on the ‘Lytro-05’ multi-focus image pair.

# Acknowledgement
The authors acknowledge the National Natural Science Foundation of China (Grant nos. 61772319, 62002200, 61976125, 61976124 and 12001327), and Shandong Natural Science Foundation of China (Grant no. ZR2017MF049).
