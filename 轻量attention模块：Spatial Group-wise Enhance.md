## 轻量attention模块：Spatial Group-wise Enhance

李翔 [极市平台](javascript:void(0);) *昨天*

> 加入极市专业CV交流群，与**6000+来自腾讯，华为，百度，北大，清华，中科院**等名企名校视觉开发者互动交流！更有机会与**李开复老师**等大牛群内互动！
>
> 同时提供每月大咖直播分享、真实项目需求对接、干货资讯汇总，行业技术交流**。**点击文末“**阅读原文**”立刻申请入群~



作者：李翔

来源：https://zhuanlan.zhihu.com/p/66928045

已获作者授权，请勿二次转载。



![img](https://mmbiz.qpic.cn/mmbiz_png/gYUsOT36vfqBiaicyD7SsKvbBBBcUZ48THBicfFFLLYuXlLNfHk280LgCO0vPcFubpeOzYkTicfGuc0XQoM5qW7kcg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

论文地址：https://arxiv.org/pdf/1905.09646.pdf



本篇是轻量attention模块的系列之作，它的一个重要的亮点就是同时几乎不增加参数量和计算量的情况下也能让分类与检测性能得到极强的增益。同时，与其他attention模块相比，它是首个利用local与global的相似性作为attention mask的generation source，同时具有非常强的语义表示增强的可解释性。



一句话来概括本文的技术：在每个特征语义组内，利用local与global feature的相似性来指导增强语义特征的空间分布。我们来看核心技术图：



![img](https://mmbiz.qpic.cn/mmbiz_png/gYUsOT36vfqBiaicyD7SsKvbBBBcUZ48THibsibTF9aIWqNVBNibX4hDmnUib9KyQrnMEoEfmFxsNNOspRoeQBMyqfKg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)







作者为了尽可能让模块轻量，同时又要达到建模目的可谓是煞费苦心，基本上每加一个操作都必须小心翼翼，最后merge出一个基本达到最简的表达，具体操作如下：



首先将特征分组，每组feature在空间上与其global pooling后的feature做点积（相似性）得到初始的attention mask，在对该attention mask进行减均值除标准差的normalize，并同时每个组学习两个缩放偏移参数使得normalize操作可被还原，然后再经过sigmoid得到最终的attention mask并对原始feature group中的每个位置的feature进行scale。



每个SGE模块引入大约2倍组数个参数，组数一般在32或64，这个数量级基本在大几十。相比于百万级别的CNN而言基本上**参数量的增加基本可忽略不计。**



这么设计的出发点也很容易理解，我们希望能够增强CNN学到的feature的语义分布，使得在正确语义的region，特征能够突出，而在无关语义的region，特征向量能够尽可能接近0。概念上受Capsule等启发，首先我们将特征分组，并认为每组特征在学习地过程中能够捕捉到某一个特定的语义。自然地，我们可以将global的平均feature代表该组学习到的语义向量（至少是接近的，否则该组就都被noise feature dominate了，那我做不做操作都没太大影响）。接下来，我们用每个position的feature与该global feature做点积，那么根据点积的定义，那些本身模长大的feature以及与global feature向量方向接近的feature就会得到一个较大的初始attention mask数值，这也是我们所期望的。因为不同样本在同一组上求得的attention mask分布差异很大，所以我们需要归一到同样的范围来给出准确的attention。最后，每一个location的feature都会scale上最终的0-1之间的数值。该方法的名称也准确地反应了核心操作：我们是group-wise地在spatial上enhance了语义feature的分布。



接下来看这个操作是否真的增强了分布。作者用模长代表响应值，将SGE-ResNet50在第4个stage的图像plot出来：



![img](https://mmbiz.qpic.cn/mmbiz_png/gYUsOT36vfqBiaicyD7SsKvbBBBcUZ48THCT9AVgpocu3SfYFt821benB3oltzSFLeesOFsUGzwAHXZ0DPgMxqnA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





反正当时将feature对比画出的时候我是震惊了的，尽管只有label的监督，CNN的确非常精准地学习到了一些语义特征，如狗的鼻子，舌头，耳朵，眼睛等等。而且，被SGE增强后的feature map能够更加精准地凸显这些语义区域，完全达到了建模预期的效果。令人惊叹的是，4,7行连闭眼的眼睛SGE都能很好地给capture住。



同时，在最高语义层上，SGE在ImageNet validation set所有50k样本上的统计分布完全符合我们的建模预期：更大的variance，较大的激活保留，较小的激活向0偏移：



![img](https://mmbiz.qpic.cn/mmbiz_png/gYUsOT36vfqBiaicyD7SsKvbBBBcUZ48THBnE79xicAoia2iapgIWCkguVVHB8xEjoGIMG34raKiaIsVyibnUdNibibVq3g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



![img](https://mmbiz.qpic.cn/mmbiz_png/gYUsOT36vfqBiaicyD7SsKvbBBBcUZ48TH60PibZs27tMTLmibzibdmWxEWww97vvkhgYoO76CzxgojicX1BKh2Ey7WA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





接下来就是一波在ImageNet和COCO上的实验了，为了公平比较，统一用pytorch的框架实现，每个实验都是现跑的。在ImageNet上，用ResNet50，ResNet101做backbone，与state-of-the-art的attention module比，性价比非常可观。



![img](https://mmbiz.qpic.cn/mmbiz_jpg/gYUsOT36vfqBiaicyD7SsKvbBBBcUZ48THHgKnnwD9pHde6x61Vze8PNiaPndVDvKItVGPmKlps9p8mSjJQcdorOg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





Ablation study告诉我们3个knowledge：



![img](https://mmbiz.qpic.cn/mmbiz_png/gYUsOT36vfqBiaicyD7SsKvbBBBcUZ48TH9l3E6ZRfJaubDRPAicMT3I2nDzGuQKRpMHkLsN2J61ttNicsN8z9Z99w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



\1. Group参数取适中，Top-1性能可到最高;



\2. 初始化建议将缩放的参数初始化为0，目的是在attention起到作用前先让网络脱离attention自己学一会，先学习到一个基本的semantic的表示，然后学着学着缩放参数经过梯度下降变成非0之后再渐渐使得attention起到作用；



\3. Normalization非常必要，不可去除；



最后是COCO上的实验，首先是在Two-Stage detector上的增益在1~2% AP，相当可观，同时SGE竟然还表现出了遇强则强的状态，在Cascade RCNN的较高的ResNet101的baseline上还能涨接近2个AP：



![img](https://mmbiz.qpic.cn/mmbiz_jpg/gYUsOT36vfqBiaicyD7SsKvbBBBcUZ48THTmJpmwpxtQMao3dlpbJ8kMXL6zcfUnaQ9NrETsFugsFREHcR849x0g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



在one-stage的RetinaNet上，在保持media和large与最强的attention module接近的情况下，small object的增益超过了SE/SK 1个点以上，可见其对小区域的空间分布增强带来了非常大的好处。



![img](https://mmbiz.qpic.cn/mmbiz_jpg/gYUsOT36vfqBiaicyD7SsKvbBBBcUZ48THcpyicrlb9Tc9C2G2val55fvKRwkicqc9T446Jib1A7kGxwRLicCicqZCvibw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



最后，我们的代码和模型将会陆续放在这里：

https://github.com/implus/PytorchInsight



并且，我们计划把这个工程打造成一个方便大家高效Research的地方，不断更新最新的模型、方法及技术，欢迎star，fork使用~



\------------------------------------------------------------

更新：cbam flops 计算小bug更新；detection实验中backbone的Param. Flops严格按照detection设定计算（去掉fc层，使用1333 x 800 的图forward）。











***延伸阅读**

- [Rocket Training: 一种提升轻量网络性能的训练方法](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247489258&idx=1&sn=3b32f2bb97df8c0f922407c245d3d0cb&chksm=ec1ffb13db687205b008f31d2d31cbfc42a47c947d231303b46ee2c902464809d22a9ebbf269&scene=21#wechat_redirect)
- [南邮提出实时语义分割的轻量级网络：LEDNET，可达 71 FPS！70.6% class mIoU！即将开源](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247488761&idx=1&sn=6099792f79049ceebb4ed764bb77abd8&chksm=ec1ff900db687016902eaf3746303aa22415e0278b05cd0b6cf93507d18591ccc6dc0a6cad70&scene=21#wechat_redirect)
- [CVPR2019 | 全景分割：Attention-guided Unified Network](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247487527&idx=1&sn=2f798d1036a3bd1b5c6939669fc7f557&chksm=ec1ffddedb6874c85cbbe1d700fb34f8af7d18b3c92c29fb7f30fb69910c1a665059b60663f2&scene=21#wechat_redirect)



------

点击左下角**“阅读原文”，**即可申请加入极市**目标跟踪、目标检测、工业检测、人脸方向、视觉竞赛等技术交流群，**更有每月大咖直播分享、真实项目需求对接、干货资讯汇总，行业技术交流，一起来让思想之光照的更远吧~



![img](https://mmbiz.qpic.cn/mmbiz_jpg/gYUsOT36vfqlnAWoRicbkC6cKCSmX7mzOPibdxpaj0ib3OxFHDWGibRiaQibRX18PhLiblNczf9he0uuqyNrVz9LfTZmQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



**觉得有用麻烦给个在看啦~**  **![img](https://mmbiz.qpic.cn/mmbiz_gif/gYUsOT36vfpFnEj3CMde0iaOKfGiaAmbfRRPePWld5pUR0niaibYOvNP5cx7nKS5I6180xeya4ZIYJClvHqSpQecqA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)**

[阅读原文](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247489447&idx=1&sn=b0571e1de700d15cacc213157a7f20cc&chksm=ec1ffa5edb6873481f7775aadd365aff45abcf9a8568f9e33975305531d3f60fabe48c799815&mpshare=1&scene=1&srcid=0704spoqDsbkPNwNlQu9etyQ&key=c47853a08ff0b5df2bf4e659644de3c1b306fdbefdefc280805b10a488130bc57cb55636a3a06723bb592a285077afc3e26bcc3b22f43843040d788c584c1ef1d6df4a7af7101174a3103b973e18bc36&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=tFNqUL0VfHxxY99IyVywfi5SR9hyyWsrjaXd5I2BiPMy%2BpgePcB11%2FXQntJivQur##)





![img](https://mp.weixin.qq.com/mp/qrcode?scene=10000004&size=102&__biz=MzI5MDUyMDIxNA==&mid=2247489447&idx=1&sn=b0571e1de700d15cacc213157a7f20cc&send_time=)

微信扫一扫
关注该公众号