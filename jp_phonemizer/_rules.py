import re
from ._data import (
    all_transformations,
    formal_to_informal,
    formal_to_informal2,
    formal_to_informal3,
    k_mapper,
    mapper,
    nasal_sound,
    roma_mapper,
    roma_mapper_plus_2,
    spaces,
    sym_ns,
    sym_ws,
)

def apply_transformations(text, transformations = all_transformations):
    for key, value in transformations.items():
        text = text.replace(key, value)
    return text

def Roma2IPA(text):
    orig = text

    for k, v in roma_mapper.items():
        text = text.replace(k, v)

    return text

def nasal_mapper(text):
    orig = text


    for k, v in nasal_sound.items():
        text = text.replace(k, v)

    return text

def alphabetreading(text):
    alphabet_dict = {"A": "エイ",
                "B": "ビー",
                 "C": "シー",
                 "D": "ディー",
                 "E": "イー",
                 "F": "エフ",
                 "G": "ジー",
                 "H": "エイチ",
                "I":"アイ",
                "J":"ジェイ",
                "K":"ケイ",
                "L":"エル",
                "M":"エム",
                "N":"エヌ",
                "O":"オー",
                "P":"ピー",
                "Q":"キュー",
                "R":"アール",
                "S":"エス",
                "T":"ティー",
                "U":"ユー",
                "V":"ヴィー",
                "W":"ダブリュー",
                "X":"エックス",
                "Y":"ワイ",
                "Z":"ゼッド"}
    text = text.upper()
    text_ret = ""
    for t in text:
        if t in alphabet_dict:
            text_ret += alphabet_dict[t]
        else:
            text_ret += t
    return text_ret

def replace_repeated_chars(input_string):
    result = []
    i = 0
    while i < len(input_string):
        if i + 1 < len(input_string) and input_string[i] == input_string[i + 1] and input_string[i] in 'aiueo':
            result.append(input_string[i] + 'ː')
            i += 2
        else:
            result.append(input_string[i])
            i += 1
    return ''.join(result)

def replace_chars_2(text, mapping=roma_mapper_plus_2):


    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)

    pattern = '|'.join(re.escape(key) for key in sorted_keys)


    def replace(match):
        key = match.group(0)
        return mapping.get(key, key)

    return re.sub(pattern, replace, text)

def replace_tashdid_2(s):
    vowels = 'aiueoɯ0123456789.?!_。؟？！．．．＠@＃#＄$％%＾^＆&＊*（)(）_+=[「」]></\`~～―ー∺"'
    result = []

    i = 0
    while i < len(s):
        if i < len(s) - 2 and s[i].lower() == s[i + 2].lower() and s[i].lower() not in vowels and s[i + 1] == ' ':
            result.append('ʔ')
            result.append(s[i + 2])
            i += 3
        elif i < len(s) - 1 and s[i].lower() == s[i + 1].lower() and s[i].lower() not in vowels:
            result.append('ʔ')
            result.append(s[i + 1])
            i += 2
        else:
            result.append(s[i])
            i += 1

    return ''.join(result)

def hira2ipa(text, roma_mapper=roma_mapper):
    keys_set = set(roma_mapper.keys())
    special_rule = ("n", "ɴ")

    transformed_text = []
    i = 0

    while i < len(text):
        if text[i] == special_rule[0]:
            if i + 1 == len(text) or text[i + 1] not in keys_set:
                transformed_text.append(special_rule[1])
            else:
                transformed_text.append(text[i])
        else:
            transformed_text.append(text[i])

        i += 1

    return ''.join(transformed_text)

def post_fix(text):
    orig = text

    for k, v in k_mapper.items():
        text = text.replace(k, v)

    return text

def random_sym_fix(text):
    orig = text

    for k, v in sym_ws.items():
        text = text.replace(k, f" {v} ")

    return text

def random_sym_fix_no_space(text):
    orig = text

    for k, v in sym_ns.items():
        text = text.replace(k, f" {v} ")

    return text

def fix_wagas(text):

    pattern = r'\b(gaːɽɯ|waːɽɯ|gaːrɯ|waːrɯ|gaːɽi|waːɽi|gaːri|waːri)\b(?!\s*\w)'

    def replace_match(match):
        word = match.group(1)
        if 'ː' in word:
            return word.replace('ː', ' a')
        return word

    return re.sub(pattern, replace_match, text)
