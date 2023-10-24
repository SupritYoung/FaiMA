import stanza
import numpy as np

# 下载英语模型
stanza.download('en')

class SentenceAnalyzer:
    def __init__(self):
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=False)
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
        doc = self.nlp(sentence)
        n = len(doc.sentences[0].words)
        pos_matrix = np.zeros((n, n), dtype=int)

        for i, word_i in enumerate(doc.sentences[0].words):
            for j, word_j in enumerate(doc.sentences[0].words):
                pos_matrix[i, j] = self.pos2id[f"{word_i.upos}-{word_j.upos}"]

        return pos_matrix

    def dependency_parsing(self, sentence):
        doc = self.nlp(sentence)
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


def sigmoid(distance, scale=0.1):
    """
    Use sigmoid to calculate similarity score based on distance.
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


def get_center_word(sentences, center_words):
    # 找出 center word 在句子中的索引，注意可能是词组
    center_word_indices = []
    has_multi_words = False
    for i, sentence in enumerate(sentences):
        if any(len(word.split(' ')) > 1 for word in center_words[i]):
            center_word_indices.append([[sentence.split().index(w) for w in word.split()] for word in center_words[i]])
            has_multi_words = True
        else:
            center_word_indices.append([sentence.split().index(word) for word in center_words[i]])

    analyzer = SentenceAnalyzer()
    sentences_features = analyzer.batch_encode_sentences(sentences)

    # 如果有中心词包含多个 token
    if has_multi_words:
        for i, center_word_index in enumerate(center_word_indices):
            for j, word_index in enumerate(center_word_index):
                if isinstance(word_index, list):
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


def linguistic_feature(sentences, outputs, task, threshold=0.25):
    if task == 'ASPE':
        center_words = [[output[0]] for output in outputs]
    elif task == 'ASQP':
        # TODO
        center_words = [[output[0], output[2]] for output in outputs]
    else:
        AssertionError('Invalid task name')

    sentences_features, center_word_indices = get_center_word(sentences, center_words)
    similarity = calculate_similarity(sentences_features, center_word_indices)

    # 生成语言特征 (0-1 矩阵)
    linguistic_feature = np.zeros_like(similarity)
    for i in range(len(similarity)):
        for j in range(len(similarity)):
            if similarity[i][j] > threshold:
                linguistic_feature[i][j] = 1
    return linguistic_feature

if __name__ == "__main__":
    # 示例
    analyzer = SentenceAnalyzer()
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "He loves reading books."
    ]
    res = analyzer.batch_encode_sentences(sentences)

    print(res)

    
