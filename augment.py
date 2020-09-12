import json
from tqdm import tqdm, trange
import random
import secrets
try:
    from google.cloud import translate
    client = translate.Client()
except:
    print("Meant to work on Google Cloud / Colab environment! Translation function won't work")

from nltk.corpus import wordnet


def translate(text, lang='de', probability=0.5):
    if random.random() > probability:
        # the statistics were against you
        return None

    # Translate En -> Lang
    target = lang
    translation = client.translate(text, target_language=target)
    trans_output = translation["translatedText"].replace('&#39;', "'")
    trans_output = trans_output.replace('&quot;', "'")

    # Translate Lang -> En
    target = 'en'
    translation_en = client.translate(trans_output, target_language=target)
    tmp_out = translation_en["translatedText"].replace('&#39;', "'").replace('&quot;', "'")
    tmp_out = tmp_out.encode('utf8')

    return tmp_out


def add_nmt_questions_to_dataset(path_to_squad_json, prob, lang, output_path):
    with open(path_to_squad_json, encoding='utf-8') as f:
        data = json.load(f)

    data_it = trange(len(data["data"]), desc="Data", leave=False)
    for i in data_it:
        para_it = trange(len(data["data"][i]["paragraphs"]), desc="Paragraph", leave=False)
        for j in para_it:
            qas_it = trange(len(data["data"][i]["paragraphs"][j]["qas"]), desc="Question", leave=False)
            for k in qas_it:
                question = data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                translated = translate(question, lang=lang, probability=prob)
                if translated:
                    # Generate new entry
                    dict = data["data"][i]["paragraphs"][j]["qas"][k]
                    dict["id"] = secrets.token_hex(15)
                    dict["question"] = translated.decode('utf8').replace("'", '"')
                    # Append the entry to the question list
                    data["data"][i]["paragraphs"][j]["qas"].append(dict)
                qas_it.update()
            para_it.update()
        data_it.update()

    print(f'Writing NMT augmented JSON to {output_path}')
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile)


def get_synonym_sentence(sentence, probability=0.2):
    words = sentence.split(" ")
    for i in range(len(words)):
        if random.random() > probability:
            # the statistics were against you
            continue

        # search for synonym
        for syn in wordnet.synsets(words[i]):
            syn_name = syn.lemmas()[0].name()
            if syn_name != words[i]:
                words[i] = syn_name

    # return the augmented sentence
    return " ".join(words)


def add_synonym_questions(path_to_squad_json, prob, output_path):
    with open(path_to_squad_json, encoding='utf-8') as f:
        data = json.load(f)

    total_augmented = 0
    data_it = trange(len(data["data"]), desc="Data", leave=False)
    for i in data_it:
        para_it = trange(len(data["data"][i]["paragraphs"]), desc="Paragraph", leave=False)
        for j in para_it:
            qas_it = trange(len(data["data"][i]["paragraphs"][j]["qas"]), desc="Question", leave=False)
            for k in qas_it:
                question = data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                synonymed = get_synonym_sentence(question, probability=prob)
                # Add question to DB only if it is different from the original
                if question != synonymed:
                    # Generate new entry
                    total_augmented += 1
                    dict = data["data"][i]["paragraphs"][j]["qas"][k]
                    dict["id"] = secrets.token_hex(15)
                    dict["question"] = synonymed.replace("'", '"')
                    # Append the entry to the question list
                    data["data"][i]["paragraphs"][j]["qas"].append(dict)
                qas_it.update()
            para_it.update()
        data_it.update()

    print(f'Total augmented question: {total_augmented}')
    print(f'Writing synonymed augmented JSON to {output_path}')
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    pass
    # add_synonym_questions("./data/train-v2.0.json", prob=0.2, output_path="./data/train-v2_syn20.json")
    # add_nmt_questions_to_dataset("./data/train-v2.0.json", prob=0.35, lang="de", output_path="./data/train-v2_de35.json")