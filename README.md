 [自蒸馏技术在小模型训练中的应用研究.docx](https://github.com/user-attachments/files/22949988/default.docx)
作者黄佳豪 莆田擢英中学初中部 C231116
# 🧠 Self-Distillation Project — 自蒸馏技术在小模型训练中的应用研究

本项目为论文《自蒸馏技术在小模型训练中的应用研究》的配套开源代码与实验记录。  
我们通过在小型卷积神经网络上引入自蒸馏（Self-Distillation）机制，实现了模型泛化性能的提升，而无需引入更大的教师模型。  

---

## 📌 研究背景

传统的知识蒸馏（Knowledge Distillation）依赖一个庞大的教师模型来指导小模型的学习。  
而**自蒸馏**跳过了教师模型这一步，让学生“向自己学习”，大幅简化了蒸馏流程，并起到正则化作用。  

✅ 优势包括：
- 🚀 提升小模型泛化性能  
- ⚡ 不增加推理开销  
- 🧠 缓解过拟合  
- 📊 易于部署与迁移  

---

## 🧪 实验配置

- **数据集**：CIFAR-10  
- **模型结构**：ResNet-18（小模型）  
- **训练轮数**：30 epochs  
- **优化器**：SGD (momentum=0.9, weight_decay=5e-4)  
- **蒸馏温度 T**：4  
- **损失函数**：交叉熵 + KL 散度蒸馏损失  
- **蒸馏权重 α**：0.5  

---

---

## 📊 实验结果

| 模型                | Top-1 准确率 | 验证损失 | 训练-验证精度差 |
|---------------------|-------------|----------|-----------------|
| Baseline            | 81.7%       | 0.573    | ≈6%             |
| Self-Distillation   | 82.1%       | 0.538    | ≈4%             |

- 自蒸馏模型在验证集上准确率提高了 **0.4%**
- 损失更低，训练曲线与验证曲线更接近，泛化能力更强  
- 在猫狗、鹿马等易混类别上的识别准确率略有提升

---

## 📈 可视化结果

### Baseline 训练曲线
![Baseline Accuracy](baseline_curves_acc.png)
![Baseline Loss](baseline_curves_loss.png)

### 自蒸馏训练曲线
![SelfDistill Accuracy](selfdistill_curves_acc.png)
![SelfDistill Loss](selfdistill_curves_loss.png)

### 混淆矩阵对比
| Baseline | Self-Distillation |
|----------|-------------------|
| ![Baseline Confmat](baseline_confmat_best.png) | ![SelfDistill Confmat](selfdistill_confmat_best.png) |

---

## 📜 引用 / Citation

如果你在研究中使用了本项目，请引用以下论文：[自蒸馏技术在小模型训练中的应用研究.docx](https://github.com/user-attachments/files/22950019/default.docx)
---

## 📎 数据与代码可用性

本项目所有实验代码、训练日志（`history.csv`）、绘图脚本与中间结果均已公开，  
👉 仓库地址：[https://github.com/JiahaoAlbus/selfdistill-project](https://github.com/JiahaoAlbus/selfdistill-project)

---

## 📬 联系方式

如需合作或进一步交流，欢迎联系作者：  
**Email:** jiahao15345932820@gmail.com  
**GitHub:** [@JiahaoAlbus](https://github.com/JiahaoAlbus)

---

⭐ 如果你觉得这个项目有帮助，欢迎 **Star** 一下支持我！
---

🧠 *“Teach yourself, become your own teacher.”*
