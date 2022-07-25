# studyAI

### 特征模板

跟踪目标再原图中对应的区域块。该块包含了目标的表面特征。通过计算模板和原图的像素差(相似度)判断对应关系。

## 全局跟踪

滑动模板，全局搜索原图中的每个像素，来寻找需要跟踪的目标。`MatchTemplate`

1. 通过设置滑动步长、像素步长来减少计算量、提高运行速度。
2. 全局搜索能够对跟丢目标进行二次锁定。但耗时较大。

## 局部跟踪

1. 考虑目标时空连续性，再规定的范围内进行搜索，耗时较少。`MatchTemplate`
2. 当目标逃离搜索框将无法进行二次锁定。
3. 对于机动性较大的目标 搜索框应设计得较大。
4. 每次估计下一帧搜索范围 `EstimateSearchArea`

## 表面特征更新

1. 永不更新表面特征（模板）
2. 每帧更新，与时俱进，漂移严重。T(K+1)=T(k)
3. 一阶滤波更新。T(k+1)=aT(K)+bT(k-1).下一帧模板由这一帧和上一帧模板共同决定。当a较大时，模板更新

   `cv::addWeighted(this->TargetTemplate, this->params.alpha, this->CurrentTargetPatch, 1.0 - this->params.alpha, 0.0, this->TargetTemplate);`


* `MatchTemplateMatchTemplatethis->CurrentTargetPatch由this->TargetTemplate` `匹配后产生-->T(k)`
* `this->TargetTemplate 产生CurrentTargetPatch` 的模板-->T(k-1)

## 随机采样特征匹配

1. 采样点数量和计算量相关，不同的模板计算量不同导致计算时间相差较大。
2. 采样集合应该重视模板中心，弱化模板边缘。
3. 正态分布随机采样 `GenerateRandomSamplePoints`
4. 首先产生定量的随机数，根据模板的大小缩放随机样本的位置。

## 图像尺度变化

* 解决不同尺度的目标物与模板大小不匹配的问题

1. 产生不同尺度的模板  `GenerateMultiScaleTargetTemplates`
2. 在局部区域中进行多模版匹配。匹配过程中调整模板 `MatchMultiScaleTemplates`
3. 模板库中，根据产生的当前帧识别结果产生相同大小的模板 `UpdateMultiScaleTemplates`
