<div align="center">
  <div align="center">
    <h1><b>📊 表格结构识别</b></h1>
  </div>
</div>

## 内容列表
- [简介](##简介)
- [安装](##安装)
- [使用](##使用)
- [公开使用的数据集](##数据集)
- [未来的工作](##未来的工作)

## 简介
💖该仓库是用来对文档中表格做结构化识别的推理库，表格结构使用的是自训练模型。相关代码是基于[TableStructureRec](https://github.com/RapidAI/TableStructureRec/tree/main) 进行的二次开发。

## 安装
``` python {linenos=table}
pip install -r requirements.txt
```

## 使用
我们提供了可直接使用的简单脚本，一键即可输出三种格式，包括html，json结果和中间结果的可视化。
``` 
python inference_batch.py --input_folder test_image
```

## 未来的工作
- 结合最新自研的去扭曲与文本矫正模型
- 优化模型对复杂无线表的表现
- 公开模型的训练脚本与数据
