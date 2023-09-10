# 日期

## 2023.09.08

# 论文标题

## [Faster R-CNN Towards Real-Time Object]([1506.01497.pdf (arxiv.org)](https://arxiv.org/pdf/1506.01497.pdf))

# 摘要

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations.Advances like SPPnet [1] and Fast R-CNN [2] have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features—using the recently popular terminology of neural networks with ’attention’ mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model [3], our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.

# 引用信息（BibTeX格式）

```BibTeX
 @article{Ren_He_Ross_Jian_2016,  
   title={Faster R-CNN Towards Real-Time Object}, 
   url={http://dx.doi.org/10.1109/TPAMI.2016.2577031}, 
   DOI={10.1109/TPAMI.2016.2577031}, 
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
   author={Shaoqing Ren; Kaiming He; Ross Girshick; Jian Sun}, 
   year={2016}, 
   month={June}, 
   pages={1137 - 1149} 
 }
```

# 本论文解决什么问题

解决了传统的计算机视觉领域中目标检测算法的**准确性和速度之间的矛盾**,实现更高的检测速度。

# 已有方法的优缺点

传统的目标检测算法包括以下几种：

1. 基于滑动窗口的方法：这类算法通过在图像上**以不同的尺度和位置**滑动一个**固定大小**的窗口，对每个窗口进行分类器的预测，来判断是否存在目标。常见的方法有**基于特征描述子**（？？？）的方法，如HOG（Histogram of Oriented Gradients）+ SVM（Support Vector Machine），以及**基于Haar特征**的级联分类器等。

2. 基于区域提议的方法：这类算法**首先生成潜在的目标区域候选框**，然后对这些候选框进行**分类和定位**。常见的方法有Selective Search、EdgeBoxes和R-CNN（Region-based Convolutional Neural Networks）等。

3. 基于特征金字塔的方法：这类算法通过**构建多尺度的特征金字塔**，从而在不同尺度下有效地检测目标。常见的方法有SIFT（Scale-Invariant Feature Transform）和SURF（Speeded-Up Robust Features）等。

4. 基于模板匹配的方法：这类算法**将目标与已知的模板进行匹配**，从而完成目标检测。常见的方法有基于模板匹配的相关滤波器（Correlation Filter）和基于形状匹配的方法等。

这些传统的目标检测算法在一定程度上取得了成功，但在**准确性和速度之间存在**着一定的**矛盾**。

他们各自的**优缺点**：

1. 滑动窗口方法：

- 优点：简单直观，易于实现。
- 缺点：计算量大，需要在不同尺度和位置进行滑动窗口，效率较低。对目标尺寸变化较大的情况不适用。

2. 区域提议方法：

- 优点：候选框生成准确，能够处理目标尺寸变化较大的情况。

- 缺点：算法复杂度较高，需要额外的区域提议步骤，导致整体速度较慢。

3. 特征金字塔方法：

- 优点：能够检测到不同尺度的目标，具有较好的尺度不变性。

- 缺点：计算量较大，需要构建多尺度的特征金字塔，导致速度较慢。

4. 模板匹配方法：

- 优点：简单快速，适用于特定场景中目标形状和外观变化较小的情况。

- 缺点：对目标形状和外观的变化敏感，对光照、遮挡等因素的影响较大。不适用于复杂场景和目标变化较大的情况。

# 本文采用什么方法及其优缺点

"Faster R-CNN"提出了一种**端到端**的深度学习框架，能够在**不牺牲准确性**的前提下实现更高的检测速度。它引入了一种称为"**区域提议网络**"（Region Proposal Network, RPN）的组件，通过**共享卷积特征**并同时进行目标分类和边界框回归来生成候选区域。这个创新的设计使得**整个目标检测过程**可以在一个**统一的框架中**进行训练和推断，从而大大提高了检测的效率和准确性。

Faster R-CNN网络的图像处理过程：

![image-20230908095129628](https://files.catbox.moe/50qm90.png)

图像处理流程如下：

![image-20230908095129628](https://files.catbox.moe/xs8d3u.png)



Faster-RCNN可以采用多种的**主干特征提取网络**（Backbone），常用的有VGG，Resnet，Xception等等。如果以Resnet为例，Faster R-CNN对输入的图片尺寸没有固定，但是**一般会把短边固定成**600，输入一张1200x1800的图片，会把图片**不失真**的resize到600x900上。

ResNet50有两个基本的块，分别名为**Conv Block**和**Identity Block**，其中Conv Block输入和输出的**维度**是**不一样**的，所以不能连续串联，它的作用是**改变网络的维度**；Identity Block输入维度和输出**维度相同**，可以串联，用于**加深网络**的。

Faster-RCNN的主干特征提取网络只包含了长宽压缩了四次的内容，第五次压缩后的内容在ROI中使用。最后一层的输出就是**公共特征层**（**FeatureMap**）。

**公共特征层**有两个应用：

- 和ROIPooling结合使用

- 进行一次3x3的卷积后，进行一个18通道的1x1卷积，还有一个36通道的1x1卷积。

作者提出了**anchor**的概念，num_priors也就是先验框的**数量是9**，所以1x1卷积的结果就是，9 x 4的卷积用于预测公用特征层上每一个网格点上**每一个先验框的变化情况**。9 x 2的卷积用于预测公用特征层上每一个网格点上**每一个预测框内部是否包含了物体**，序号为1的内容为包含物体的概率。通过这一步获得Proposal建议框。

利用建议框对公共特征层进行**截取**，并将截取的结果进行**Resize**和下一步的卷积。从而获取最终的预测结果并进行解码。

作者还提出了LossFunction、NMS、归一化处理等概念，用来测试新模型的实验效果。

其中比较**难懂**的有**4-Step Alternating Training**、**Ablation Experiments on RPN**、**mean Average Precision(mAP)**、**Sensitivities to Hyper-parameters**。

还通过对比实验，比较了本模型和其他模型的效果。

Faster R-CNN的**优缺点**：

- 优点：准确性高、**端到端**的训练、**可扩展**性（应用于不同尺寸、不同类别、不同领域）、较快的检测速度。
- 缺点：计算**资源要求高**、**模型复杂度高**、较长的训练时间和较高的内存消耗、难以处理**小目标和密集目标**。

# 使用的数据集和性能度量

Faster R-CNN**没有指定的特定数据集**，它可以应用于各种不同的目标检测任务和数据集，PASCAL VOC 2007 test set、PASCAL VOC 2012 test set、MS COCO dataset、ImageNet等都进行了测试。通过平均精度**mean Average Precision(mAP)**进行性能的度量。

测试方式和相关结果如下：

1. ZF网络在PASCAL VOC 2007测试集上每个锚点的**平均候选框**大小

![image-20230908150103337](https://files.catbox.moe/jfei6u.png)

实验目的：有助于更好地**理解和解释模型的性能**，并可以指导后续的优化和改进。





2. 在PASCAL VOC 2007 test set和VOC 2007 train set上的效果

![image-20230908150415127](https://files.catbox.moe/finoam.png)

目的是比较使用不同提议方法的**Fast R-CNN with ZF模型的性能**。这些方法可能会影响模型的召回率、准确率和速度等因素。

以及消融实验...





3. 同样的测试集，训练集是PASCAL VOC 2007和PASCAL VOC 2012，检测器为Fast R-CNN和VGG-16

![image-20230908150701774](https://files.catbox.moe/q32azw.png)

实验目的：

- 评估模型性能--->Fast R-CNN和VGG-16模型。

- 比较不同训练数据的影响--->两种不同的训练数据集进行实验，分别是"07"（VOC 2007 trainval)和"07+12"（VOC 2007 trainval和VOC 2012 trainval的并集）。

- 研究RPN的影响--->评估其在Fast R-CNN模型中的效果，以及该参数对目标检测性能的影响。





4. 测试集是PASCAL VOC 2012 test set，Training data: “07”: VOC 2007 trainval, “07++12”: union set of VOC 2007 trainval+test and VOC 2012 trainval，检测器为Fast R-CNN和VGG-16

![image-20230908151434850](https://files.catbox.moe/m5l16k.png)

实验目的：与第三大致相同...





5. 评估在K40 GPU上使用Fast R-CNN和VGG-16模型进行目标检测时的运行时间

   ![image-20230908151943190](https://files.catbox.moe/98dogx.png)

使用了**不同部分的时间统计**，包括**SS候选框生成方法**在CPU上的评估时间以及**区域合并**、**池化**、**全连接**和**softmax**等层的时间。**目的**是评估模型的运行时间、比较不同模块的时间开销。



6. 测试集是PASCAL VOC 2007 test set和PASCAL VOC 2012 test set，检测器为Fast R-CNN和VGG-16，比较了不同物体的检测的mAP![image-20230908152444796](https://files.catbox.moe/gbguu4.png)

实验目的：比较RPN和RPN∗的效果...



7. 测试**锚的尺寸和纵横比、超参数以及建议框的数量**对平均精度mAP的影响

   ![image-20230908152937425](https://files.catbox.moe/h4j6ht.png)



8. **one-stage detection vs Two-Stage Proposal + Detection**

![image-20230908153136486](https://files.catbox.moe/i4nh8l.png)





9. 更换不同的数据集，**MS COCO dataset**，模型是VGG-16，测试效果...

   ![image-20230908153356547](https://files.catbox.moe/kg0u5s.png)





10. 测试不同的训练集，VOC07、VOC07+12、VOC07++12、COCO(no VOC)、COCO+VOC07+12、COCO+VOC07++12。模型是VGG-16,测试集是PASCAL VOC 2007和PASCAL VOC 2012。

    ![image-20230908154343154](https://files.catbox.moe/05a178.png)



# 与我们工作的相关性

Faster R-CNN是一种**目标检测算法**，它引入了**区域建议网络**（Region Proposal Network，RPN）来生成**候选目标框**，并结合了Fast R-CNN进行**目标分类和边界框回归**。

# 英文总结

"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" is a research paper published in 2015 by Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun in the IEEE Transactions on Pattern Analysis and Machine Intelligence. The paper presents the Faster R-CNN algorithm, which introduces the Region Proposal Network (RPN) for generating candidate object proposals and combines it with Fast R-CNN for object classification and bounding box regression.

The authors' key innovations in this paper include:

1. Introducing the Region Proposal Network (RPN): The RPN generates region proposals by sliding a small window over the feature map and predicting potential object locations and their associated scores. This eliminates the need for external region proposal methods and significantly improves the efficiency of object detection.
2. Unified framework for region proposals and object detection: Faster R-CNN integrates the RPN with Fast R-CNN, creating a unified framework that performs both region proposal generation and subsequent object classification and bounding box refinement. This end-to-end approach simplifies the overall detection pipeline and increases detection accuracy.

The experimental methodology employed by the authors involves training and evaluation on benchmark datasets such as PASCAL VOC and MS COCO. They demonstrate that the Faster R-CNN algorithm achieves state-of-the-art performance in terms of both accuracy and speed compared to previous object detection methods.

In conclusion, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" introduces the RPN as a novel approach for generating region proposals, effectively integrating it with Fast R-CNN. Through extensive experiments, the authors demonstrate the algorithm's superior performance in terms of accuracy and real-time object detection capabilities. This paper has significantly influenced the field of object detection and continues to serve as a fundamental reference for subsequent research and advancements in the field.