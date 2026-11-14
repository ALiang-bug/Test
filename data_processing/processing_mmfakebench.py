import re
import json
import os

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}


def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_data(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def expand_contractions(text: str) -> str:

    lower_to_orig = {}
    words = re.split(r'(\W+)', text)
    lower_words = []
    for w in words:
        lw = w.lower()
        lower_words.append(lw)
        lower_to_orig[lw] = w        


    pattern_parts = []
    replacement_map = {}          

    for contr, full in contractions.items():
        if contr in lower_to_orig: 

            pattern_parts.append(rf'\b{re.escape(contr)}\b')
            replacement_map[contr] = full

    if not pattern_parts:
        return text 

    big_regex = re.compile('|'.join(pattern_parts), flags=re.IGNORECASE)

    def repl(match):
        orig_word = match.group(0)
        lower_word = orig_word.lower()
        target = replacement_map.get(lower_word, orig_word)

        if orig_word.isupper():
            return target.upper()
        elif orig_word.istitle():
            return target.capitalize()
        else:
            return target

    return big_regex.sub(repl, text)

def finalize_text(text: str) -> str:

    text = expand_contractions(text)

    text = text.strip()
    if not text.endswith('.'):
        text += '.'
    return text


def process_single_item(item, json_file_path):
    json_dir_name = os.path.splitext(os.path.basename(json_file_path))[0]

    original_image_path = item.get('image_path', 'unknown')
    if original_image_path != 'unknown':
        if original_image_path.startswith('/'):
            original_image_path = original_image_path[1:]
        image_path = f"{json_dir_name}/{original_image_path}"
        item['image_path'] = image_path
    else:
        image_path = 'unknown'

    try:
        item['text'] = finalize_text(item['text'])
        return item
    except Exception as e:
        print(f"Error: {e}, image_path: {image_path}")
        return item

def process_data(train_data, output_file, json_file_path):
    processed_data = [process_single_item(it, json_file_path) for it in train_data]
    write_data(processed_data, output_file)
    return len(processed_data)

if __name__ == "__main__":
    train_json_path = '../dataset/MMFakeBench/MMFakeBench_test.json'
    test_json_path  = '../dataset/MMFakeBench/MMFakeBench_val.json'

    print("Currently, the training data is being read...")
    train_data = read_data(train_json_path)
    print(f"The training data has been loaded successfully. In total, {len(train_data)}")

    print("Currently, the test data is being read...")
    test_data = read_data(test_json_path)
    print(f"The test data has been loaded successfully. In total, {len(test_data)}")

    print("\nStart processing the training data...")
    train_count = process_data(
        train_data,
        "MMFakeBench/train_set.json",
        train_json_path
    )
    print(f"The training data processing is completed. The data has been successfully processed and saved {train_count}")

    print("\nStart processing the test data...")
    test_count = process_data(
        test_data,
        "MMFakeBench/test_set.json",
        test_json_path
    )
    print(f"The test data processing is completed. The data has been successfully processed and saved {test_count}")

    print("\nAll data processing is complete!")