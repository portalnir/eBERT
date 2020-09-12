# Meant to work on Google Cloud / Colab environment
import json
from tqdm.notebook import tqdm, trange
import random
import secrets
from google.cloud import translate
client = translate.Client()


def translate(text, lang='de', probability=0.5):
    if random.random() < probability:
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
    else:
        # the statistics were against you
        return None

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

    print(f'Writing augmented JSON to {output_path}')
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    add_nmt_questions_to_dataset("./data/train-v2.0.json", prob=0.35, lang="de", output_path="./data/train-v2de_35.json")