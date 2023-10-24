import numpy as np


def cosine_similarity(v1, v2):
    """计算两个向量之间的余弦相似度"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def compute_sentiment_similarity(inputs, outputs, task, threshold=0.5):
    """计算情感向量之间的相似度矩阵"""
    # 处理成一个二维 list，统计 positive, neutral, negative 的数量
    if task == 'ASPE' or task == 'MEMD_AS':
        sentiment_vectors = [[0, 0, 0] for _ in range(len(inputs))]
        for i, output in enumerate(outputs):
            for o in output:
                sentiment = o[1]
                if sentiment == 'positive' or sentiment == 'POS':
                    sentiment_vectors[i][1] += 1
                elif sentiment == 'negative' or sentiment == 'NEG':
                    sentiment_vectors[i][0] += 1
                elif sentiment == 'neutral' or sentiment == 'NEU':
                    sentiment_vectors[i][2] += 1
                else:
                    raise ValueError(f"Invalid sentiment: {sentiment}")
    # 三元组和四元组抽取任务 添加对于 implicit aspect 和 Opinion 的考虑        
    elif task == 'MEMD_AOS':
        # 第 4 维 表示隐式 opinion
        sentiment_vectors = [[0, 0, 0, 0] for _ in range(len(inputs))]
        for i, output in enumerate(outputs):
            for o in output:
                if o[2] == 'NULL':
                    sentiment_vectors[i][3] += 1

                sentiment = o[1]
                if sentiment == 'positive' or sentiment == 'POS':
                    sentiment_vectors[i][1] += 1
                elif sentiment == 'negative' or sentiment == 'NEG':
                    sentiment_vectors[i][0] += 1
                elif sentiment == 'neutral' or sentiment == 'NEU':
                    sentiment_vectors[i][2] += 1
                else:
                    raise ValueError(f"Invalid sentiment: {sentiment}")
    elif task == 'MEMD_ACOS':
        # 第 5 维 表示隐式 aspect
        sentiment_vectors = [[0, 0, 0, 0, 0] for _ in range(len(inputs))]
        for i, output in enumerate(outputs):
            for o in output:
                if o[3] == 'NULL':
                    sentiment_vectors[i][3] += 1
                if o[0] == 'NULL':
                    sentiment_vectors[i][4] += 1

                sentiment = o[2]
                if sentiment == 'positive' or sentiment == 'POS':
                    sentiment_vectors[i][1] += 1
                elif sentiment == 'negative' or sentiment == 'NEG':
                    sentiment_vectors[i][0] += 1
                elif sentiment == 'neutral' or sentiment == 'NEU':
                    sentiment_vectors[i][2] += 1
                else:
                    raise ValueError(f"Invalid sentiment: {sentiment}")
    else:
        raise ValueError(f"Invalid task: {task}")

    n = len(sentiment_vectors)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity = cosine_similarity(sentiment_vectors[i], sentiment_vectors[j])
            similarity = (similarity + 1) / 2  # 将相似度从[-1, 1]映射到[0, 1]
            if similarity > threshold:
                similarity_matrix[i][j] = 1
            else:
                similarity_matrix[i][j] = 0

    return similarity_matrix

def compute_domain_similarity(inputs, domains):
    """计算领域相似度矩阵"""
    n = len(inputs)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if domains[i] == domains[j]:
                similarity_matrix[i][j] = 1

    return similarity_matrix


def euclidean_distance(v1, v2):
    """计算两个向量之间的欧几里得距离"""
    return np.linalg.norm(np.array(v1) - np.array(v2))

def compute_structural_similarity(outputs, task, threshold=0.3):
    """计算结构相似度矩阵，主要考虑方面、观点长度、隐式与否"""
    similarity_matrix = []
    if task == 'ASPE' or task == 'MEMD_AS':
        for output in outputs:
            implicit_aspect, aspect_length = 0, 0
            for o in output:
                if o[0] == 'NULL':
                    implicit_aspect += 1
                aspect_length += len(o[0].split(' '))
            similarity_matrix.append([implicit_aspect, aspect_length])
        # TODO
    else:
        raise ValueError(f"Invalid task: {task}")

    n = len(similarity_matrix)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance = euclidean_distance(similarity_matrix[i], similarity_matrix[j])
            similarity = 1 / (1 + distance)
            if similarity > threshold:
                similarity_matrix[i][j] = 1
            else:
                similarity_matrix[i][j] = 0
    
    return similarity_matrix


if __name__ == "__main__":
    # 示例
    # sentiment_vectors = [[2, 1, 0], [0, 0, 2]]
    # print(compute_similarity_matrix(sentiment_vectors))

    data = [[2, 7, 1, 3], [3, 8, 0, 2]]
    print(compute_structural_similarity(data))
