import re
import cutlet

katsu = cutlet.Cutlet(ensure_ascii=False)
katsu.use_foreign_spelling = False
for symbol in ('♡', '♪', '○', 'っ', 'ッ', '◆'):
    katsu.update_mapping(symbol, symbol)


def merge_katakana_phonemes(output, original_text):

    text_without_punct = re.sub(r'[!?！？。、,.\s]', '', original_text)
    katakana_chars = len(re.findall(r'[\u30A0-\u30FF]', text_without_punct))
    total_chars = len(text_without_punct)

    if total_chars == 0 or (katakana_chars / total_chars > 0.8):

        return output


    if not re.search(r'[\u30A0-\u30FF]', original_text):

        return output


    pattern = r'\b([a-z]{1,3}ː)\s+([a-z]{1,3}(?:ː)?[a-z]{0,2})\b'

    def should_merge(match):
        part1, part2 = match.groups()

        if len(part1) <= 4 and len(part2) <= 4 and 'ː' in (part1 + part2):
            return True
        return False


    result = output
    for match in re.finditer(pattern, result):
        if should_merge(match):
            original_match = match.group(0)
            merged = match.group(1) + match.group(2)
            result = result.replace(original_match, merged, 1)

    return result


def is_english_text(text):

    core_text = re.sub(r'[\s\.,!?\-\'\"]+', '', text)
    if not core_text:
        return False


    english_chars = len(re.findall(r'[A-Za-z0-9]', core_text))
    total_chars = len(core_text)


    if total_chars > 0 and english_chars / total_chars > 0.8:
        return True


    if re.search(r'[A-Z]', text):
        return True

    return False


def split_japanese_english(text):
    segments = []
    current_segment = ""
    current_is_english = None


    i = 0
    while i < len(text):


        english_word_match = re.match(r'[A-Za-z]+[\'\-]?[A-Za-z]*', text[i:])

        if english_word_match:
            word = english_word_match.group()


            if current_is_english is False and current_segment:
                segments.append((current_segment, False))
                current_segment = ""

            current_is_english = True
            current_segment += word
            i += len(word)


            while i < len(text) and text[i] in ' \t\n.,!?;:':
                current_segment += text[i]
                i += 1


                if text[i-1] in '。、！？':
                    segments.append((current_segment, True))
                    current_segment = ""
                    current_is_english = None

        else:

            char = text[i]


            if (ord(char) > 127 or char in '。、！？「」『』（）'):
                if current_is_english is True and current_segment:

                    segments.append((current_segment.rstrip(), True))
                    current_segment = ""

                current_is_english = False
                current_segment += char
            else:

                if current_segment:
                    current_segment += char
                else:

                    current_segment = char
                    current_is_english = False

            i += 1


    if current_segment:
        segments.append((current_segment, current_is_english if current_is_english is not None else False))


    if not segments:
        segments = [(text, False)]


    merged_segments = []
    for segment, is_eng in segments:
        if merged_segments and merged_segments[-1][1] == is_eng:

            merged_segments[-1] = (merged_segments[-1][0] + segment, is_eng)
        else:
            merged_segments.append((segment, is_eng))

    return merged_segments
