aspe_icl_templates = [
'''Now let's do an ASPE (Aspect-Sentiment Pair Extraction) task. Task definition: The output will be the aspects (clearly appear in the text) and the aspects sentiment polarity (positive, neutral, negative). I will give you some example.

{example}
Now please complete the ASPE task for the following input:\n"{input}"\nOutput:\n''',

'''Now let's do an ASPE (Aspect-Sentiment Pair Extraction) task. Task definition: Your aim is to identify explicit aspects and determine their sentiment polarity (positive, neutral, negative). For clarity, here are some examples:

{example}
Now, using the above examples as guidance, perform the ASPE task for the sentence below:\n"{input}"\nOutput:\n''',

'''Let's dive into the ASPE task (Aspect-Sentiment Pair Extraction). Here's what you need to do: Extract aspects, and determine their sentiment polarities, which can be positive, neutral, or negative. I've provided some examples to guide you:

{example}
Using the examples as a reference, apply the ASPE procedure to the following input:\n"{input}"\nOutput:\n''',

'''It's time for the ASPE (Aspect-Sentiment Pair Extraction) task. Your objective is to extract aspects, and categorize their sentiment polarities as positive, neutral, or negative. Here are some instances for your reference:

{example}
Keeping the examples as a baseline, evaluate the next input via the ASPE procedure:\n"{input}"\nPlease give the extraction results:\n''',

'''For the ASPE task, identify aspects and their sentiment polarities (positive, neutral, negative). Consider these examples to grasp the essence:

{example}
Now, based on the understanding, evaluate the following sentence:\n"{input}"\nOutput:\n''',

'''Your next task is ASPE (Aspect-Sentiment Pair Extraction). The goal is to detect aspects and their sentiment polarities which can be either positive, negative, or neutral. Some examples you can learn for reference are below:

{example}
Ready? Let's apply this to the sentence below:\n"{input}"\nGive me your output:\n''',

'''Help me complete an aspect sentiment pair extraction task: locate and extract explicit aspects mentioned in the sentence and classify their sentiment as either positive, negative, or neutral. Provide both the aspect and its corresponding sentiment polarity. Refer to the example below:

{example}
Now, tackle the following input:\n"{input}",Can you tell me what your results are?\n''',

'''I need you to do an ASPE (Aspect-Sentiment Pair Extraction) task. The goal is to extract aspects and their corresponding sentiment polarities (positive, neutral, negative). Here are some examples:

{example}
Now, based on the examples and its format, evaluate the following sentence:\n"{input}"\nThe ASPE task results:\n''',

'''I'm feeling overwhelmed with an ASPE (Aspect-Sentiment Pair Extraction) task. Can you pitch in and help? The task is aimed at extracting aspects and their corresponding sentiment polarities from the given sentences. Aspects will be explicitly mentioned, and sentiment polarities can only be positive, negative, or neutral. Here are some templates you can refer to:

{example}
Now, I'm handing this task over to you. The input sentences are:\n"{input}"\nWhat are your results?\n''',

'''Could you please lend me a hand with an ASPE (Aspect-Sentiment Pair Extraction) task? The task's goal is to find and extract aspects explicitly mentioned in the sentence, and then determine their associated sentiment polarities as positive, negative, or neutral. Report both the aspect and its sentiment polarity. I will give you some templates to learn how to extract.

{example}
Now, it's your show time. The input sentences are:\n"{input}"\nWhat is your output?\n'''
]

aspe_templates = [
'''Now let's do an ASPE (Aspect-Sentiment Pair Extraction) task. Task definition: The output will be the aspects (clearly appear in the text) and the aspects sentiment polarity (positive, neutral, negative).
Now please complete the ASPE task for the following input:\n"{input}"\nOutput:\n''',

'''Now let's do an ASPE (Aspect-Sentiment Pair Extraction) task. Task definition: Your aim is to identify explicit aspects and determine their sentiment polarity (positive, neutral, negative).
Now, using the above examples as guidance, perform the ASPE task for the sentence below:\n"{input}"\nOutput:\n''',

'''Let's dive into the ASPE task (Aspect-Sentiment Pair Extraction). Here's what you need to do: Extract aspects, and determine their sentiment polarities, which can be positive, neutral, or negative.
Using the examples as a reference, apply the ASPE procedure to the following input:\n"{input}"\nOutput:\n''',

'''It's time for the ASPE (Aspect-Sentiment Pair Extraction) task. Your objective is to extract aspects, and categorize their sentiment polarities as positive, neutral, or negative.
Keeping the examples as a baseline, evaluate the next input via the ASPE procedure:\n"{input}"\nPlease give the extraction results:\n''',

'''For the ASPE task, identify aspects and their sentiment polarities (positive, neutral, negative).
Now, based on the understanding, evaluate the following sentence:\n"{input}"\nOutput:\n''',

'''Your next task is ASPE (Aspect-Sentiment Pair Extraction). The goal is to detect aspects and their sentiment polarities which can be either positive, negative, or neutral.
Ready? Let's apply this to the sentence below:\n"{input}"\nGive me your output:\n''',

'''Help me complete an aspect sentiment pair extraction task: locate and extract explicit aspects mentioned in the sentence and classify their sentiment as either positive, negative, or neutral. Provide both the aspect and its corresponding sentiment polarity.
Now, tackle the following input:\n"{input}",Can you tell me what your results are?\n''',

'''I need you to do an ASPE (Aspect-Sentiment Pair Extraction) task. The goal is to extract aspects and their corresponding sentiment polarities (positive, neutral, negative).
Now, based on the examples and its format, evaluate the following sentence:\n"{input}"\nThe ASPE task results:\n''',

'''I'm feeling overwhelmed with an ASPE (Aspect-Sentiment Pair Extraction) task. Can you pitch in and help? The task is aimed at extracting aspects and their corresponding sentiment polarities from the given sentences. Aspects will be explicitly mentioned, and sentiment polarities can only be positive, negative, or neutral.
Now, I'm handing this task over to you. The input sentences are:\n"{input}"\nWhat are your results?\n''',

'''Could you please lend me a hand with an ASPE (Aspect-Sentiment Pair Extraction) task? The task's goal is to find and extract aspects explicitly mentioned in the sentence, and then determine their associated sentiment polarities as positive, negative, or neutral. Report both the aspect and its sentiment polarity.
Now, it's your show time. The input sentences are:\n"{input}"\nWhat is your output?\n'''
]

aste_icl_templates = [
'''Now let's do an ASTE (Aspect-Sentiment Triple Extraction) task. Task definition: The output will be the aspects (clearly appear in the text), the aspects opinion (may not be present in the text, if not be present, please replace it with 'NULL'), and the aspects sentiment polarity (positive, neutral, negative). I will give you some examples.

{example}
Now please complete the ASTE task for the following input:\n"{input}"\nOutput:\n''',

'''Now let's do an ASTE (Aspect-Sentiment Triple Extraction) task. Task definition: Your aim is to identify explicit aspects, extract the opinions of the aspects (if not exist in the sentence, please replace the position with 'NULL'), and determine their sentiment polarity (positive, neutral, negative). For clarity, here are some examples:

{example}
Now, using the above examples as guidance, perform the ASTE task for the sentence below:\n"{input}"\nOutput:\n''',

'''Let's dive into the ASTE (Aspect-Sentiment Triple Extraction) task. Here's what you need to do: Extract aspects, identify the opinions corresponding to the aspects (if not exist, can be implicitly represented as 'NULL'), and determine their sentiment polarity, which can be positive, neutral, or negative. I've provided some examples to guide you:

{example}
Using the examples as a reference, apply the ASTE procedure to the following input:\n"{input}"\nOutput:\n''',

'''It's time for the ASTE (Aspect-Sentiment Triple Extraction) task. Your objective is to extract aspects, find the opinions of these aspects (if can't find, you can represent them as 'NULL'), and categorize their sentiment polarity as positive, neutral, or negative. Here are some instances for your reference:

{example}
Keeping the examples as a baseline, evaluate the next input via the ASTE procedure:\n"{input}"\nPlease give the extraction results:\n''',

'''Now, let's begin an ASTE task, for ASTE task, you need to identify aspects, opinions (you can replace them with 'NULL' if you can't), and their sentiment polarity (positive, neutral, negative). Consider these examples to grasp the essence:

{example}
Now, based on the understanding, evaluate the following sentence:\n"{input}"\nOutput:\n''',

'''Your next task is ASTE (Aspect-Sentiment Triple Extraction). The goal is to detect explicit aspects, opinions (please replace with 'NULL' if can't), and their sentiment polarities (positive, neutral, negative) from given sentences. Some examples are below:

{example}
Ready? Let's apply this to the sentences below:\n"{input}"\nGive me your output:\n''',

'''Help me complete an aspect sentiment triple extraction task: Extract fine-grained aspects, opinions (you can replace them with 'NULL' if not be present), and their respective sentiment polarities (positive, neutral, negative) from the sentences. Be attentive to nuanced aspect-opinion pairs and ensure the sentiment polarities' accuracy. You can refer to the example below:

{example}
Now, tackle the following input:\n"{input}"\nPlease give me your output''',

'''I need you to do an ASTE (Aspect-Sentiment Triple Extraction) task. The goal is to extract aspects, the opinion of each of them (if not exist, please replace it with 'NULL'), and their sentiment polarity (positive, neutral, negative). Here are some examples you can learn:

{example}
Now, based on the examples and its format, evaluate the following sentence:\n"{input}"\nThe ASTE task results:\n''',

'''Suppose you are a researcher in the field of NLP actively tackling the ASTE (Aspect-Sentiment Triple Extraction) task. Your task is to extract the aspects from the given sentences and their corresponding opinions and sentiment polarities (positive, neutral, negative). Be attentive to replace opinion's position with 'NULL' if cannot detect. Below are some templates for your reference:

    {example}
Now, you have learned the provided templates, please start tackling the given sentences! The pending sentences are below:\n"{input}"\nWhat is your output\n''',

'''I need you to help me complish an ASTE (Aspect-Sentiment Triple Extraction) task. Next, you need to extract aspects and corresponding opinions from sentences, meanwhile, ensure that the sentiment polarity (positive, neutral, negative) is accurately labeled for each opinion. If an aspect lacks an opinion, use 'NULL' to replace the opinion. Here are some templates you can learn:

{example}
Next, please begin your task based on the templates provided above. The sentences you input are:\n"{input}"\nplease tell me what is your output?\n''',

'''I am working on an ASTE task in the NLP field, and I need your assistance.Please help me extract the aspect, corresponding opinion (use "NULL" if no opinion is expressed), and assign one of the sentiment polarities: "positive," "negative," or "neutral" for each given sentence. Ensure that you include all aspects, even if their corresponding opinions are empty, and consider the three sentiment polarities only: positive, negative, and neutral. Next, I will provide you with some examples for your reference.

{example}
Now, please begin, the sentences input are:\n"{input}" what are your results?\n''',

'''Hello, I need you! your task is to extract aspects, opinions (use "NULL" if no opinion exists), and assign sentiment polarities ("positive," "negative," or "neutral") for each sentence provided. Please ensure that all aspects are identified, even if their corresponding opinions are empty, and limit the sentiment polarities to positive, negative, and neutral. Some templates are below:

{example}
Now, it's your show time, the sentences input are:\n"{input}" please give me your output\n'''
]

aste_templates = [
'''Now let's do an ASTE (Aspect-Sentiment Triple Extraction) task. Task definition: The output will be the aspects (clearly appear in the text), the aspects opinion (may clear appear or not be present in the text, if not be present, please replace it with 'NULL'), and the aspects sentiment polarity (positive, neutral, negative).
Now please complete the ASTE task for the following input:\n"{input}"\nOutput:\n''',

'''Now let's do an ASTE (Aspect-Sentiment Triple Extraction) task. Task definition: Your aim is to identify explicit aspects, extract the opinions of the aspects (if not exist in the sentence, please replace the position with 'NULL'), and determine their sentiment polarity (positive, neutral, negative).
Now, using the above examples as guidance, perform the ASTE task for the sentence below:\n"{input}"\nOutput:\n''',

'''Let's dive into the ASTE (Aspect-Sentiment Triple Extraction) task. Here's what you need to do: Extract aspects, identify the opinions corresponding to the aspects (if not exist, can be implicitly represented as 'NULL'), and determine their sentiment polarity, which can be positive, neutral, or negative.
Using the examples as a reference, apply the ASTE procedure to the following input:\n"{input}"\nOutput:\n''',

'''It's time for the ASTE (Aspect-Sentiment Triple Extraction) task. Your objective is to extract aspects, find the opinions of these aspects (if can't find, you can represent them as 'NULL'), and categorize their sentiment polarity as positive, neutral, or negative.
Keeping the examples as a baseline, evaluate the next input via the ASTE procedure:\n"{input}"\nPlease give the extraction results:\n''',

'''Now, let's begin an ASTE task, for ASRE task, you need to identify aspects, opinions (you can replace them with 'NULL' if you can't), and their sentiment polarity (positive, neutral, negative).
Now, based on the understanding, evaluate the following sentence:\n"{input}"\nOutput:\n''',

'''Your next task is ASTE (Aspect-Sentiment Triple Extraction). The goal is to detect explicit aspects, opinions (please replace with 'NULL' if can't), and their sentiment polarities (positive, neutral, negative) from given sentences.
Ready? Let's apply this to the sentences below:\n"{input}"\nGive me your output:\n''',

'''Help me complete an aspect sentiment triple extraction task: Extract fine-grained aspects, opinions (you can replace them with 'NULL' if not be present), and their respective sentiment polarities (positive, neutral, negative) from the sentences. Be attentive to nuanced aspect-opinion pairs and ensure the sentiment polarities' accuracy.
Now, tackle the following input:\n"{input}"\nPlease give me your output''',

'''I need you to do an ASTE (Aspect-Sentiment Triple Extraction) task. The goal is to extract aspects, the opinion of each of them (if not exist, please replace it with 'NULL'), and their sentiment polarity (positive, neutral, negative).
Now, based on the examples and its format, evaluate the following sentence:\n"{input}"\nThe ASTE task results:\n''',

'''Suppose you are a researcher in the field of NLP actively tackling the ASTE (Aspect-Sentiment Triple Extraction) task. Your task is to extract the aspects from the given sentences and their corresponding opinions and sentiment polarities (positive, neutral, negative). Be attentive to replace opinion's position with 'NULL' if cannot detect.
Now, you have learned the provided templates, please start tackling the given sentences! The pending sentences are below:\n"{input}"\nWhat is your output?\n''',

'''I need you to help me complish an ASTE (Aspect-Sentiment Triple Extraction) task. Next, you need to extract aspects and corresponding opinions from sentences, meanwhile, ensure that the sentiment polarity (positive, neutral, negative) is accurately labeled for each opinion. If an aspect lacks an opinion, use 'NULL' to replace the opinion.
Next, please begin your task based on the templates provided above. The sentences you input are:\n"{input}"\nplease tell me what is your output?\n''',

'''I am working on an ASTE task in the NLP field, and I need your assistance.Please help me extract the aspect, corresponding opinion (use "NULL" if no opinion is expressed), and assign one of the sentiment polarities: "positive," "negative," or "neutral" for each given sentence. Ensure that you include all aspects, even if their corresponding opinions are empty, and consider the three sentiment polarities only: positive, negative, and neutral.
Now, please begin, the sentences input are:\n"{input}" what are your results?\n''',

'''Hello, I need you! your task is to extract aspects, opinions (use "NULL" if no opinion exists), and assign sentiment polarities ("positive," "negative," or "neutral") for each sentence provided. Please ensure that all aspects are identified, even if their corresponding opinions are empty, and limit the sentiment polarities to positive, negative, and neutral.
Now, it's your show time, the sentences input are:\n"{input}" please give me your output\n'''
]

asqp_icl_templates = [
'''Now let's do an ASQP (Aspect-Sentiment Quad Prediction) task. Task definition: The output will be the aspects along with their corresponding categories, opinions, and sentiment polarities (positive, neutral, negative). But you need to pay attention to two points: Firstly, Aspect and Opinion may not appear in the sentences, in such cases, please replace them with 'NULL'. Secondly, the determination of the Category should be chosen from the given set of categories, and you should learn the category set from the provided examples. Some examples are below.

{example}
Now please complete the ASQP task for the following input:\n"{input}"\nOutput:\n''',

'''Now let's do an ASQP (Aspect-Sentiment Quad Prediction) task. Task definition: Your aim is to identify aspects (if the sentence does not contain it, please replace it with NULL) and determine their categories (you need to learn how to determine the category from the given examples.), opinions (if the sentence does not contain it, please replace it with NULL) and sentiment polarities (positive, neutral, negative). For clarity, here are some examples:

{example}
Now, using the above examples as guidance, perform the ASQP task for the sentence below:\n"{input}"\nOutput:\n''',

'''Let's dive into the ASQP (Aspect-Sentiment Quad Prediction) task. Here's what you need to do: Extract aspects (if not exist, can be implicitly represented as 'NULL'), identify categories (you should learn the set of categories from given examples) and opinions (if not exist, can be implicitly represented as 'NULL') corresponding to the aspects, and determine their sentiment polarity, which can be positive, neutral, or negative. I've provided some examples to guide you:

{example}
Using the examples as a reference, apply the ASQP procedure to the following input:\n"{input}"\nOutput:\n''',

'''It's time for the ASQP (Aspect-Sentiment Quad Prediction) task. Your objective is to extract aspects, categorize the aspects based on the provided examples, find the opinions of these aspects (if can't find, you can represent them as 'NULL'), and classify their sentiment polarities as positive, neutral, or negative. Here are some instances for your reference:

{example}
Keeping the examples as a baseline, evaluate the next input via the ASQP procedure:\n"{input}"\nPlease give the extraction results:\n''',

'''For the ASQP task, identify aspects (you can replace them with 'NULL' if you can't), categories (you should chose one of categories which learn from given examples), opinions (you can replace them with 'NULL' if you can't), and their sentiment polarity (positive, neutral, negative). Consider these examples to grasp the essence:

{example}
Now, based on the understanding, evaluate the following sentence:\n"{input}"\nOutput:\n''',

'''Your next task is ASQP (Aspect-Sentiment Quad Prediction). The goals are below: 
1. Aspect Extraction: Identify aspects or entities mentioned in the sentence. If no aspect is present, use "NULL" to indicate its absence.
2. Category Assignment: Determine the category associated with each identified aspect. The category set should be learned from the provided template examples.
3. Opinion Extraction: Detect any opinions expressed in the sentence related to the identified aspects. If no opinion is expressed, replace it with "NULL."
4. Sentiment Polarity Assignment: Assign one of three sentiment polarities—positive, negative, or neutral—to each opinion identified.
There are some examples for your learning:

{example}
Ready? Let's apply this to the sentences below:\n"{input}"\nGive me your output:\n''',

'''Help me complete an aspect sentiment quadruple extraction task:
1.Aspect Extraction: Identify aspects/entities, using "NULL" if none are present.
2.Category Assignment: Determine categories for each aspect from provided templates.
3.Opinion Extraction: Detect opinions, replacing with "NULL" if absent.
4.Sentiment Polarity: Assign sentiment polarity (positive, negative, neutral) to each opinion.
Ensure your annotations cover all aspects of the sentence following guidelines.
Refer to the examples below, you can learn and draw inspiration.

{example}
Now, tackle the following input:\n"{input}"\nPlease give me your output''',

'''Suppose you are a researcher in the field of NLP actively tackling the ASQP (Aspect-Sentiment Quad Prediction) task. Your task is to extract the aspects from the given sentences and their corresponding categories, opinions and sentiment polarities (positive, neutral, negative). However, please be mindful of the following: 
Firstly, aspects and opinions may not always be present in the sentences. If they are missing, use 'NULL' to replace the aspect or opinion. 
Secondly, the categories for aspects come from a predefined set, but this set needs to be learned from the provided template examples.
Below are some templates for your reference:

{example}
Now, you have learned the provided templates, please start tackling the given sentences! The pending sentences are below:\n"{input}"\nWhat is your output?\n'''
]

asqp_templates = [
'''Now let's do an ASQP (Aspect-Sentiment Quad Prediction) task. Task definition: The output will be the aspects along with their corresponding categories, opinions, and sentiment polarities (positive, neutral, negative). But you need to pay attention to one point: Aspect and Opinion may not appear in the sentences, in such cases, please replace them with 'NULL'.
Now please complete the ASQP task for the following input:\n"{input}"\nOutput:\n''',

'''Now let's do an ASQP (Aspect-Sentiment Quad Prediction) task. Task definition: Your aim is to identify aspects (if the sentence does not contain it, please replace it with NULL) and determine their categories (you need to infer which category each fine-grained aspect belongs to on your own), opinions (if the sentence does not contain it, please replace it with NULL) and sentiment polarities (positive, neutral, negative).
Now, using the above examples as guidance, perform the ASQP task for the sentence below:\n"{input}"\nOutput:\n''',

'''Let's dive into the ASQP (Aspect-Sentiment Quad Prediction) task. Here's what you need to do: Extract aspects (if not exist, can be implicitly represented as 'NULL'), identify categories (you should deduce the category for each fine-grained aspect yourself) and opinions (if not exist, can be implicitly represented as 'NULL') corresponding to the aspects, and determine their sentiment polarity, which can be positive, neutral, or negative.
Using the examples as a reference, apply the ASQP procedure to the following input:\n"{input}"\nOutput:\n''',

'''It's time for the ASQP (Aspect-Sentiment Quad Prediction) task. Your objective is to extract aspects, categorize the aspects based on you common sense, find the opinions of these aspects (if can't find, you can represent them as 'NULL'), and classify their sentiment polarities as positive, neutral, or negative.
Keeping the examples as a baseline, evaluate the next input via the ASQP procedure:\n"{input}"\nPlease give the extraction results:\n''',

'''For the ASQP task, identify aspects (you can replace them with 'NULL' if you can't), categories (you need to deduce on your own which category each fine-grained aspect belongs to), opinions (you can replace them with 'NULL' if you can't), and their sentiment polarity (positive, neutral, negative).
Now, based on the understanding, evaluate the following sentence:\n"{input}"\nOutput:\n''',

'''Your next task is ASQP (Aspect-Sentiment Quad Prediction). The goals are below: 
1. Aspect Extraction: Identify aspects or entities mentioned in the sentence. If no aspect is present, use "NULL" to indicate its absence.
2. Category Assignment: Determine the category associated with each identified aspect. However, the category set needs to be formulated by you based on your own common sense.
3. Opinion Extraction: Detect any opinions expressed in the sentence related to the identified aspects. If no opinion is expressed, replace it with "NULL."
4. Sentiment Polarity Assignment: Assign one of three sentiment polarities—positive, negative, or neutral—to each opinion identified.
Ready? Let's apply this to the sentences below:\n"{input}"\nGive me your output:\n''',

'''Help me complete an aspect sentiment quadruple extraction task:
1.Aspect Extraction: Identify aspects/entities, using "NULL" if none are present.
2.Category Assignment: Determine categories for each aspect from your common sense.
3.Opinion Extraction: Detect opinions, replacing with "NULL" if absent.
4.Sentiment Polarity: Assign sentiment polarity (positive, negative, neutral) to each opinion.
Ensure your annotations cover all aspects of the sentence following guidelines.
Now, tackle the following input:\n"{input}"\nPlease give me your output''',

'''Suppose you are a researcher in the field of NLP actively tackling the ASQP (Aspect-Sentiment Quad Prediction) task. Your task is to extract the aspects from the given sentences and their corresponding categories, opinions and sentiment polarities (positive, neutral, negative). However, please be mindful of the following: 
Firstly, aspects and opinions may not always be present in the sentences. If they are missing, use 'NULL' to replace the aspect or opinion. 
Secondly, the categories for aspects come from a predefined set, but the category set needs to be defined by you based on your own knowledge.
Now, you have learned the provided templates, please start tackling the given sentences! The pending sentences are below:\n"{input}"\nWhat is your output?\n'''
]