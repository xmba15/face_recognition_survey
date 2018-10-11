# Face Recognition Survey
* [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
    * CVPR 2015
    * Triplet Lossを用いた128次元の顔画像embeddingの学習
    * 学習中、オンラインでTriplet(Hard NegativeとHard Positive)を選択
    * [Source1](https://github.com/davidsandberg/facenet)
    * [Source2](https://github.com/cmusatyalab/openface)
    * [Source3](https://github.com/shanren7/real_time_face_recognition)
* [MobileID: Face Model Compression by Distilling Knowledge from Neurons](http://personal.ie.cuhk.edu.hk/~pluo/pdf/aaai16-face-model-compression.pdf)
    * AAAI 2016
    * [Source](https://github.com/liuziwei7/mobile-id)
* [Triplet Probabilistic Embedding for Face Verification and Clustering](https://arxiv.org/pdf/1604.05417.pdf)
    * BTAS 2016
    * [Source](https://github.com/meownoid/face-identification-tpe)
* [A Discriminative Feature Learning Approach for Deep Face Recognition](http://ydwen.github.io/papers/WenECCV16.pdf)
    * ECCV 2016
    * Softmax Loss + Center Loss
    * [Source](https://github.com/pangyupo/mxnet_center_loss)
    * [Source](https://github.com/ydwen/caffe-face)
* [Pose-Robust Face Recognition via Deep Residual Equivariant Mapping](https://arxiv.org/pdf/1803.00839.pdf)
    * CVPR 2018
    * 横顔を正面顔にMappingするDREAM Blockの導入により、顔の姿勢にロバストな顔認識
    * [Source](https://github.com/penincillin/DREAM)
* [MobileFaceNets: Efficient CNNs for Accurate RealTime Face Verification on Mobile Devices](https://arxiv.org/pdf/1804.07573.pdf)
    * CCBR 2018
    * MobileNetV2を元にした顔画像embedding学習
    * 携帯電話向けの早いモデル
    * [Source](https://github.com/sirius-ai/MobileFaceNet_TF)
* [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
    * [Source](https://github.com/deepinsight/insightface#deep-face-recognition)
* [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf)
    * CVPR 2018
    * LossをCosine距離で計算する際、クラス間にmarginをつける
    * OSSにはLicense付けられてない
    * [Source](https://github.com/yule-li/CosFace)

# Hashing high-dimensional vector
* [FAst Lookups of Cosine and Other Nearest Neighbors](https://falconn-lib.org/)
    * [Source](https://github.com/falconn-lib/falconn)

* [Locality Sensitive Hashing using MinHash]
    * [Source](https://github.com/mattilyra/LSH)
* [Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk]
    * Spotifyが開発した音楽推薦システムのためのApproximate Nearest Neighbors
    * [Source](https://github.com/spotify/annoy)s

