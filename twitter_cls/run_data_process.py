"""
@file   : run_data_process.py
@time   : 2025-12-28
"""
import re
from tqdm import tqdm
import pandas as pd


def clean_data(texts):
    cleaned_texts = []
    for text in tqdm(texts):
        """
        保留情感信息的清洗
        """
        # 2. 移除转发标记（包含//@和回复@）
        text = re.sub(r'(//@|回复@|转发微博@)[\w\u4e00-\u9fa5\-_]+:', ' ', text)

        # 3. 移除单独的@提及用户
        text = re.sub(r'(?<!\[)@[\w\u4e00-\u9fa5\-_]+', ' ', text)

        # 4. ✅ 额外处理：去除单独的/（不是//@中的/）
        # 先处理特殊情况的斜杠
        text = re.sub(r'(?<!:)//(?!@)', ' ', text)  # 处理单独的//
        text = re.sub(r'(?<!/)/(?!/)', ' ', text)  # 处理单独的/，但保留//

        # 5. 移除URL链接（URL中常包含/）
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # 6. 处理各种括号（保留内容）
        # 方括号 [] - 微博表情
        text = re.sub(r'\[(.*?)\]', r'\1', text)
        # 实心括号 【】 - 中文强调
        text = re.sub(r'【(.*?)】', r'\1', text)
        # 英文括号 ()
        text = re.sub(r'\((.*?)\)', r'\1', text)
        # 中文括号 （）
        text = re.sub(r'（(.*?)）', r'\1', text)

        # 7. 处理#话题标签（保留内容）
        text = re.sub(r'#([^#]+?)#', r'\1', text)
        # 处理未闭合的话题标签
        text = re.sub(r'#([^\s#]+)', r'\1', text)

        # 8. ✅ 最后再清理所有剩余的/
        text = text.replace('/', ' ')

        # 9. 清理多余空格和特殊空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        cleaned_texts.append(text)
    return cleaned_texts


if __name__ == '__main__':
    df = pd.read_csv("./data/train.csv")

    df = df.drop(columns=['id'])

    train_df = df.sample(frac = 0.96)
    dev_df = df.drop(train_df.index)

    train_df['keyword'] = train_df['keyword'].fillna(train_df['keyword'].mode().values[0])
    train_df['location'] = train_df['location'].fillna(train_df['location'].mode().values[0])

    train_text = train_df['text'].tolist()
    cleaned_train_texts = clean_data(train_text)
    train_df['text'] = cleaned_train_texts
    train_df['text'] = train_df[['keyword', 'location', 'text']].agg(''.join, axis=1)
    train_df = train_df.drop(columns = ['keyword', 'location'])

    dev_df['keyword'] = dev_df['keyword'].fillna(dev_df['keyword'].mode().values[0])
    dev_df['location'] = dev_df['location'].fillna(dev_df['location'].mode().values[0])

    dev_text = dev_df['text'].tolist()
    cleaned_dev_texts = clean_data(dev_text)
    dev_df['text'] = cleaned_dev_texts
    dev_df['text'] = dev_df[['keyword', 'location', 'text']].agg(''.join, axis=1)
    dev_df = dev_df.drop(columns = ['keyword', 'location'])

    test_df = pd.read_csv('./data/test.csv')
    test_df['keyword'] = test_df['keyword'].fillna(test_df['keyword'].mode().values[0])
    test_df['location'] = test_df['location'].fillna(test_df['location'].mode().values[0])
    test_text = test_df['text'].tolist()
    cleaned_test_texts = clean_data(test_text)
    test_df['text'] = cleaned_test_texts
    test_df['text'] = test_df[['keyword', 'location', 'text']].agg(''.join, axis=1)
    test_df = test_df.drop(columns = ['keyword', 'location'])

    train_df.to_csv("./data/train_data.csv", index=False)
    dev_df.to_csv("./data/dev_data.csv", index=False)
    test_df.to_csv("./data/test_data.csv", index=False)


