import stanza
import numpy as np
import torch
import os
import stanza

# 设置模型路径为您上传的位置
os.environ["STANZA_RESOURCES_DIR"] = "/root/stanza_resources/"

class SentenceAnalyzer:
    def __init__(self, task, threshold=0.25):
        '''
        :param task: ABSA 子任务类型
        :param threshold: 两个句子在语言学相似与否的阈值
        '''
        self.task = task
        self.threshold = threshold
        # 需要关闭自动下载、开启预分词、开启 GPU
        if torch.cuda.is_available():
            self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=True, download_method=None, tokenize_pretokenized=True)
        else:
            self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=False, download_method=None, tokenize_pretokenized=True)
        self.pos2id = {}
        self.dep2id = {}


    def _get_unique_tags_deps(self, sentences):
        all_tags = set()
        all_deps = set()

        for sentence in sentences:
            doc = self.nlp(sentence)
            for word in doc.sentences[0].words:
                all_tags.add(word.upos)
                all_deps.add(word.deprel)

        # Create all possible combinations of pos tags
        all_combinations = {f"{tag1}-{tag2}" for tag1 in all_tags for tag2 in all_tags}

        self.pos2id = {tag: i for i, tag in enumerate(all_combinations)}
        self.dep2id = {dep: i for i, dep in enumerate(all_deps)}

    def pos_tagging(self, sentence):
        # # note 预先分词，保持统一的方法
        doc = self.nlp([sentence.split()])
        n = len(doc.sentences[0].words)
        pos_matrix = np.zeros((n, n), dtype=int)

        for i, word_i in enumerate(doc.sentences[0].words):
            for j, word_j in enumerate(doc.sentences[0].words):
                pos_matrix[i, j] = self.pos2id[f"{word_i.upos}-{word_j.upos}"]

        return pos_matrix

    def dependency_parsing(self, sentence):
        doc = self.nlp([sentence.split()])
        n = len(doc.sentences[0].words)
        dep_matrix = np.zeros((n, n), dtype=int)

        for word in doc.sentences[0].words:
            head_index = word.head - 1
            if head_index >= 0:  # Check to avoid index -1 for the root
                dep_matrix[word.id - 1, head_index] = self.dep2id[word.deprel]

        return dep_matrix

    def batch_encode_sentences(self, sentences):
        self._get_unique_tags_deps(sentences)

        s = len(sentences)
        max_n = max([len(self.nlp(sentence).sentences[0].words) for sentence in sentences])

        result = np.zeros((s, max_n, max_n, 2), dtype=int)

        for idx, sentence in enumerate(sentences):
            pos_res = self.pos_tagging(sentence)
            dep_res = self.dependency_parsing(sentence)
            result[idx, :pos_res.shape[0], :pos_res.shape[1], 0] = pos_res
            result[idx, :dep_res.shape[0], :dep_res.shape[1], 1] = dep_res

        return result

    def get_center_word(self, sentences, center_words, task):
        # 找出 center word 在句子中的索引，注意可能是词组
        center_word_indices = []

        for i, sentence in enumerate(sentences):
            sentence_indices = []
            for words_group in center_words[i]:
                # 遇到隐式 Aspect 或 Opinion 就跳过
                for word in words_group:
                    if word == 'NULL':
                        continue

                    if len(word.split(' ')) > 1:
                        sentence_words, word_tokens = sentence.split(), word.split()
                        indices = [sentence_words.index(token) for token in word_tokens if token in sentence_words]
                    else:
                        # 使用 split() 获取单词的索引
                        indices = [sentence.split().index(word)]
                    sentence_indices.append(indices)
            center_word_indices.append(sentence_indices)

        sentences_features = self.batch_encode_sentences(sentences)

        
        for i, center_word_index in enumerate(center_word_indices):
            for j, word_index in enumerate(center_word_index):
                # 如果有中心词包含多个 token
                if len(word_index) > 1:
                    max_value, max_idx = -1, None
                    # 取句法非零关系最多的 token 作为中心词
                    for idx in word_index:
                        total_count = np.count_nonzero(sentences_features[i][idx, :, 1]) + np.count_nonzero(
                            sentences_features[i][:, idx, 1]) - (1 if sentences_features[i][idx, idx, 1] != 0 else 0)
                        if total_count > max_value:
                            max_value = total_count
                            max_idx = idx
                    center_word_indices[i][j] = max_idx

        return sentences_features, center_word_indices

    def linguistic_feature(self, sentences, outputs):
        '''

        :param sentences: 原输入
        :param outputs: 原输出
        :return:
        '''
        if len(sentences) != len(outputs):
            AssertionError('The length of sentences and outputs must be the same')

        if self.task == 'ASPE' or self.task == 'MEMD_AS':
            center_words = [[[o[0]] for o in os] for os in outputs]
        elif self.task == 'MEMD_AOS':
            center_words = [[[o[0], o[2]] for o in os] for os in outputs]
        elif self.task == 'MEMD_ACOS':
            center_words = [[[o[0], o[3]] for o in os] for os in outputs]
        else:
            AssertionError('Invalid task name')

        sentences_features, center_word_indices = self.get_center_word(sentences, center_words, self.task)
        similarity = calculate_similarity(sentences_features, center_word_indices)

        # 打印相似度，统计后再确定阈值
        # print(similarity.mean(), similarity.max())
        # print(similarity)

        # 生成语言特征 (0-1 矩阵)
        linguistic_feature = np.zeros_like(similarity)
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                if similarity[i][j] > self.threshold:
                    linguistic_feature[i][j] = 1

        return linguistic_feature

def gauss_weight(center_idx, length, sigma=2.0):
    """
    Calculate the weight of each token based on its distance to the center token.
    """
    weights = np.zeros(length)
    for idx in center_idx:
        dists = np.abs(np.arange(length) - idx)
        current_weights = np.exp(-np.square(dists) / (2 * sigma ** 2))
        weights = np.maximum(weights, current_weights)
    return weights


def hamming_distance(T1, T2):
    """
    Calculate the hamming distance between two feature tensors.
    """
    return np.sum(T1 != T2)


def sigmoid(distance, scale=0.01):
    """
    Use sigmoid to calculate similarity score based on distance.
    scale 越大，函数会更陡，即对输入的微小变化反应会更敏感。
    """
    return 1 / (1 + np.exp(scale * distance))


def calculate_similarity(matrix, center_tokens):
    s, n, _, _ = matrix.shape
    similarity_matrix = np.zeros((s, s))

    # Iterating over each sentence pair
    for i in range(s):
        for j in range(s):
            if i >= j:
                continue

            center_i = center_tokens[i]
            center_j = center_tokens[j]

            weights_i = gauss_weight(center_i, n)
            weights_j = gauss_weight(center_j, n)

            weighted_matrix_i = matrix[i] * weights_i[:, np.newaxis, np.newaxis]
            weighted_matrix_j = matrix[j] * weights_j[:, np.newaxis, np.newaxis]

            min_centers = min(len(center_i), len(center_j))
            distances = []
            for c_i in center_i[:min_centers]:
                for c_j in center_j[:min_centers]:
                    dist = hamming_distance(weighted_matrix_i[c_i], weighted_matrix_j[c_j])
                    distances.append(dist)

            # We average the similarity score over the minimal center tokens
            similarity_matrix[i][j] = similarity_matrix[j][i] = np.mean([sigmoid(dist) for dist in distances])

    return similarity_matrix


if __name__ == "__main__":
    # 示例
    
    # sentences = [
    #     "GVK Power can go up to 14-15 levels : Vijay Bhambwani",
    #     "Moodys gives fifth reason for markets to cheer ; record highs seen",
    #     "Simply _ Brianah _ Hayley _ G Europe , , , Do n't ask how I know , , LoL , RT LaLaLaLove _ Jas : Where 's lady gaga , .",
    #     "Here are the things that made me confident with my purchase : Build Quality - Seriously , you ca n't beat a unibody construction .",
    #     "The 2.9ghz dual-core i7 chip really out does itself .",
    #     "Fortis Healthcare to transfer land assets to REIT ; stock down",
    #     "The food is delicious but the service is horrible.",
    #     "The phone feels ok and the battery is very durable.",
    #     "The paper quality of the book is simply amazing"
    # ]
    # center_words = [[['GVK Power', 'positive']],
    #                 [['markets', 'positive'], ['Moodys', 'neutral']],
    #                 [['lady gaga', 'neutral']],
    #                 [['Build Quality', 'positive'], ['unibody construction', 'positive']],
    #                 [['2.9ghz dual-core i7 chip', 'negative']],
    #                 [['Fortis Healthcare', 'positive'], ['REIT', 'negative']],
    #                 [["food", 'positive'], ["service", 'negative']], 
    #                 [['phone', 'positive'], ['battery', 'negative']], 
    #                 [["paper quality", 'positive']]]
    
    sentences = [
        'Recommended highly , but suggest the books in the series be read in order .',
        'It was well written and cleared up a lot of points that the media got wrong .'
    ]
    center_words = [[['NULL', 'Book#General', 'POS', 'Recommended highly']],
                    [['NULL', 'Book#Quality', 'POS', 'well']]]

    analyzer = SentenceAnalyzer(task='MEMD_ACOS')

    linguistic_feature = analyzer.linguistic_feature(sentences, center_words)
    print(linguistic_feature)