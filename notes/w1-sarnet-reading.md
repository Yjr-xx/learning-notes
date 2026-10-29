# W1 笔记 · SARNet 读懂了什么（200 字）

日期：2026-10-28
来源：老师主页 `wbwang17.github.io` 的 Publications 条目

## 正文

王老师组做的是「让计算机看懂遥感图里物体之间的关系」。场景图生成（SGG）不只是认出「这里有车、那里有停车场」，还要说出「车停在停车场里」——把物体和它们之间的关系一起输出成（主体, 关系, 客体）这样的三元组。

SARNet 要解决的麻烦是：卫星图幅面极大，一个目标和几百像素外的一栋楼其实没有关系，但常规方法会把远处目标也算进来，远处的噪声搅乱了真正该判断的关系。他们做了两件事：一是把「一个目标的影响范围」从圆改成有方向的椭圆——物体之间的影响本来就分方向，车主要影响前后、不太影响侧面；二是专门让模型多注意两个框之间的几何位置，因为「在里面」和「在旁边」的区别，靠的正是位置构型。代码已开源。

我目前的实际水平是：只会调大模型 API，视觉和 PyTorch 这两周才刚开始学。所以现在只能读懂他们在做什么。

## 四条自检

| 判据 | 结果 | 说明 |
|---|---|---|
| 有没有我解释不出来的词？ | 通过 | 「各向异性」我换成了「有方向的椭圆」；「影响传播」换成了「影响范围」 |
| 前两句非本专业的人能听懂吗？ | 通过 | 用一个「车停在停车场里」的例子解释了 |
| 有没有直接抄摘要原句？ | 通过 | 原句 "suppress noisy long-range interactions" 我写成了「远处的噪声搅乱了真正该判断的关系」 |
| 有没有一句假话或夸大？ | 通过 | 明确写了「只会调 API」，没有宣称任何能力 |

## 论文原文信息

- **题目**：Spatially Anisotropic Reasoning Network for Remote Sensing Scene Graph Generation
- **简称**：SARNet
- **作者**：Wenbin Wang, Yiheng Chen, Hang Sun, Dong Ren, Yansheng Li
- **出处**：IEEE TGRS 2026
- **摘要原文**：
  > Remote sensing images contain multiple functional subscenes, while relation semantics are strongly shaped by objects' relative spatial configurations. SARNet introduces Anisotropic Ellipse Influence Propagation (AEIP) to model object-centered spatial influence and suppress noisy long-range interactions, together with Relative Spatial Configuration Attention (RSCA) to enhance geometry-discriminative relation representations.
- **代码**：老师主页标注 "CODE release : SARNet"

