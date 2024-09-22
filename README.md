# SGLC
### [Project Page](https://neng-wang.github.io/SGLC/) | [Video](https://1drv.ms/f/c/262fa73419fbaa92/Es1dRoIbvBdMpaR3yKjnVpUB-BMc9TOvGyGpW6Rj0ri3sw?e=ZnAl9F) | [Arxiv](https://arxiv.org/abs/2407.08106)

This repo contains the implementation of our paper 

> **SGLC: Semantic Graph-Guided Coarse-Fine-Refine Full Loop Closing for LiDAR SLAM**
>
> [Neng Wang](https://github.com/neng-wang), [Xieyuanli Chen](https://github.com/Chen-Xieyuanli),  [Chenghao Shi](https://github.com/chenghao-shi), Zhiqiang Zheng, Hongshan Yu, Huimin Lu

**SGLC is a semantic graph guided full loop closing framework with robust  loop closure detection and 6-DoF poes estimation.**

### A related video

You can check it online on this  [link](https://1drv.ms/f/c/262fa73419fbaa92/Es1dRoIbvBdMpaR3yKjnVpUB-BMc9TOvGyGpW6Rj0ri3sw?e=ZnAl9F).

<div align="center">
    <a href="https://1drv.ms/f/c/262fa73419fbaa92/Es1dRoIbvBdMpaR3yKjnVpUB-BMc9TOvGyGpW6Rj0ri3sw?e=ZnAl9F" target="_blank"><img src="./pic/video_cover.png" width=100% /></a>
</div>

### Abstract

Loop closing is a crucial component in SLAM that helps eliminate  accumulated errors through two main steps: loop detection and loop pose  correction. The first step determines whether loop closing should be  performed, while the second estimates the 6-DoF pose to correct odometry drift. Current methods mostly focus on developing robust descriptors  for loop closure detection, often neglecting loop pose estimation. A few methods that do include pose estimation either suffer from low accuracy or incur high computational costs. To tackle this problem, we introduce SGLC, a real-time semantic graph-guided full loop closing method, with  robust loop closure detection and 6-DoF pose estimation capabilities.  SGLC takes into account the distinct characteristics of foreground and  background points. For foreground instances, it builds a semantic graph  that not only abstracts point cloud representation for fast descriptor  generation and matching but also guides the subsequent loop verification and initial pose estimation. Background points, meanwhile, are  exploited to provide more geometric features for scan-wise descriptor  construction and stable planar information for further pose refinement.  Loop pose estimation employs a coarse-fine-refine registration scheme  that considers the alignment of both instance points and background  points, offering high efficiency and accuracy. We evaluate the loop  closing performance of SGLC through extensive experiments on the KITTI  and KITTI-360 datasets, demonstrating its superiority over existing  state-of-the-art methods. Additionally, we integrate SGLC into a SLAM  system, eliminating accumulated errors and improving overall SLAM  performance. 

![](./pic/framework.png)



**The code  will be released after our paper  is accepted.**
