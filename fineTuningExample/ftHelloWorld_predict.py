#!/usr/bin/env python

"""
BERT Yelp Review 分类模型推理脚本
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载模型和分词器
model = AutoModelForSequenceClassification.from_pretrained("test_trainer/checkpoint-375")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# 推理函数
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predict_result = outputs.logits.argmax(dim=-1).item()
    return predict_result

if __name__ == "__main__":
    # 示例文本
    test_text = "This restaurant is amazing!"
    label = predict(test_text)
    print(f"文本: {test_text}\n预测标签: {label}")

    test_text = "This restaurant is normal."
    label = predict(test_text)
    print(f"文本: {test_text}\n预测标签: {label}")
    
    test_text = "This restaurant is bad!"
    label = predict(test_text)
    print(f"文本: {test_text}\n预测标签: {label}")