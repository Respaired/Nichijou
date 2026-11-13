import copy
import re
import cutlet
import unicodedata
import pyopenjtalk
from . import _cutlet as v1
from ._rules import (Roma2IPA, alphabetreading, apply_transformations, fix_wagas,
                     hira2ipa, nasal_mapper, post_fix, random_sym_fix,
                     random_sym_fix_no_space, replace_chars_2,
                     replace_repeated_chars, replace_tashdid_2, spaces)


_STRETCH0 = 0xE100
_STRETCH_RE = "[\ue101-\ue1ff]"
BLOCK = "\ue001"
_PROTECT = {"n'": "\ue002\ue012", "fi": "\ue003\ue013", "di": "\ue004\ue014", "si": "\ue005\ue015"}
for _c in [BLOCK] + [chr(_STRETCH0 + n) for n in range(1, 256)]:
    v1.katsu.update_mapping(_c, _c)


katsu = copy.copy(v1.katsu)
katsu.exceptions = {}

_DIGITS = re.compile(r"(?<![0-9０-９])[0-9０-９]+(?:[.,．，][0-9０-９]+)*")
_VOWEL_RUN = re.compile(r"([あいうえおアイウエオ])\1{2,}")


def _pron(node):
    pron, read = node["pron"].replace("’", ""), node["read"]
    if len(pron) != len(read):
        return pron.replace("ー", "ウ")
    return "".join(r if p == "ー" and r in "ウイ" else p for p, r in zip(pron, read))


_NUMBER_SUFFIXES = {"後", "目", "間", "代", "前", "頃", "ごろ", "ぶり", "ずつ", "おき", "余", "強", "弱", "半", "中"}
_NUMERALS = re.compile(r"[〇一二三四五六七八九十百千万億兆]+")


_SUFFIX_READING = {"後": "ゴ", "目": "メ", "間": "カン", "代": "ダイ", "前": "マエ", "頃": "ゴロ", "余": "ヨ", "半": "ハン",
                   "中": "ジュウ"}


def _nfkc(s):
    return unicodedata.normalize("NFKC", s)


def _number_spans(seg):
    spans = []
    for m in _DIGITS.finditer(seg):
        if spans and m.start() < spans[-1][1][-1][0]: continue
        rest = seg[m.end():]
        nodes = pyopenjtalk.run_frontend(m.group() + rest[:8])
        reading, k, whole = "", 0, False


        while k < len(nodes) and (nodes[k]["pos_group1"] == "数" or _NUMERALS.fullmatch(nodes[k]["string"]) or
                                  k + 1 < len(nodes) and nodes[k + 1]["pos_group1"] == "数" and k and nodes[k]["pos_group2"] == "助数詞"):
            reading += _pron(nodes[k])
            k += 1
        used = len(re.match(r"[〇一二三四五六七八九十百千万億兆]*", rest).group())

        if k < len(nodes):
            st = nodes[k]["string"]
            tail = _nfkc(re.sub(r"^[0-9０-９.,．，〇一二三四五六七八九十百千万]+", "", st))
            if tail and tail != _nfkc(st) and _nfkc(rest[used:]).startswith(tail):
                reading += _pron(nodes[k])
                used += len(tail)
                k += 1
                whole = True
        if not reading:
            continue
        number_end = m.end() + (0 if whole else used)
        whole = whole or used > 0
        counter = ""
        cands = [(m.end() + used, reading, number_end, counter, whole)]

        while (k < len(nodes) and (nodes[k]["pos_group2"] == "助数詞" or nodes[k]["string"] in _NUMBER_SUFFIXES)
               and _nfkc(rest[used:]).startswith(_nfkc(nodes[k]["string"]))):
            r = _SUFFIX_READING.get(nodes[k]["string"], _pron(nodes[k]))
            reading += r
            counter += r
            used += len(nodes[k]["string"])
            k += 1
            cands.append((m.end() + used, reading, number_end, counter, whole))

        if k < len(nodes) and nodes[k]["string"][:1] in _SUFFIX_READING and rest[used:used + 1] == nodes[k]["string"][0]:
            r = _SUFFIX_READING[nodes[k]["string"][0]]
            reading += r
            counter += r
            used += 1
            cands.append((m.end() + used, reading, number_end, counter, whole))
        spans.append((m.start(), cands))
    return spans


_SPACES_V2 = {k: v for k, v in spaces.items() if k != "i i"}


def _random_space_fix_v2(text):
    for k, v in _SPACES_V2.items():
        text = text.replace(k, v)
    return fix_wagas(text)


_VOWEL_OF = {**{c: "あ" for c in "あかさたなはまやらわがざだばぱぁゃ"}, **{c: "い" for c in "いきしちにひみりぎじぢびぴぃ"},
             **{c: "う" for c in "うくすつぬふむゆるぐずづぶぷぅゅ"}, **{c: "え" for c in "えけせてねへめれげぜでべぺぇ"},
             **{c: "お" for c in "おこそとのほもよろごぞどぼぽぉょ"}}


def _join_split_long_vowels(seg):
    words = v1.katsu.tagger(seg)
    if "".join(w.white_space + w.surface for w in words) != seg:
        return seg
    out, prev = [], None
    for w in words:
        s = w.surface
        if (prev is not None and not w.white_space and len(s) == 1 and s in "あいうえお"
                and (prev.feature.pos1 == "感動詞" or w.feature.pos1 == "記号")
                and _VOWEL_OF.get(_hira(prev.surface[-1:])) == s):
            s = "ー"
        out.append(w.white_space + s)
        prev = w
    return "".join(out)


def _stretch(n):
    return chr(_STRETCH0 + min(n, 255)) if n > 0 else ""


def _hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


_SMALL_TO_FULL = dict(zip("ぁぃぅぇぉ", "あいうえお"))


def _small_vowel(m):
    prev, small, extra, bars, nxt = m.groups()
    full = _SMALL_TO_FULL[_hira(small)]
    stretch = _stretch(len(extra) + len(bars))
    if _VOWEL_OF.get(_hira(prev)) == full:


        before_vowel = bool(re.match(r"[あいうえおアイウエオ]", nxt))
        return prev + (full if full == "う" or full == "お" and not before_vowel else "ー") + stretch
    if re.match(r"[\u4e00-\u9fff々]", prev):


        return prev + ("ー" + _stretch(len(extra) + len(bars)) if extra or bars else full)

    if extra:
        return prev + small + "ー" + _stretch(len(extra) - 1 + len(bars) or 1)
    return prev + small + bars


_LONG_PAIR_VOWELS = "アイエオ"


def _map_kana(kana):
    try:
        return katsu.map_kana(_hira(kana))
    except KeyError:
        return None


def _romaji_v2(text, numbers="digits"):
    
    if not text:
        return ""
    text = cutlet.normalize_text(text)
    words = katsu.tagger(text)
    tokens = katsu.romaji_tokens(words, False, False)
    
    prev_kana = ""
    
    for j, (w, tok) in enumerate(zip(words, tokens)):
        kana, pron = w.feature.kana or "", w.feature.pron or ""
        r = tok.surface
        cut = [i for i in range(1, len(kana)) if kana[i] in _LONG_PAIR_VOWELS
               and _VOWEL_OF.get(_hira(kana[i - 1])) == _hira(kana[i]) and len(pron) == len(kana) and pron[i] != "ー"]


        if cut and (_map_kana(kana) or "").lower() == r.lower():
            for i in reversed(cut):
                k = len(_map_kana(kana[:i]))
                r = r[:k] + BLOCK + r[k:]
        joined = j and not tokens[j - 1].space and not w.white_space
        
        if (joined and prev_kana and kana and kana[0] in _LONG_PAIR_VOWELS
                and _VOWEL_OF.get(_hira(prev_kana[-1])) == _hira(kana[0]) and pron[:1] != "ー"):
            r = BLOCK + r
        tok.surface = r
        prev_kana = kana
    _numbers_in_tokens(text, words, tokens, numbers)
    
    return "".join(str(t) for t in tokens).strip()


def _numbers_in_tokens(text, words, tokens, numbers):
    spans = _number_spans(text)
    if not spans:
        return
    bounds, pos = [], 0
    for w in words:
        pos += len(w.white_space)
        if text[pos:pos + len(w.surface)] != w.surface:
            return
        bounds.append((pos, pos + len(w.surface)))
        pos += len(w.surface)
    starts, ends = {a: j for j, (a, b) in enumerate(bounds)}, {b: j for j, (a, b) in enumerate(bounds)}
    for start, cands in spans:
        if start not in starts:
            continue
        fit = [c for c in cands if c[0] in ends and ends[c[0]] >= starts[start]]
        if not fit:
            continue
        end, reading, number_end, counter, whole = fit[-1]
        if numbers == "digits":
            if whole or not counter:
                continue
            c = _map_kana(counter)
            romaji = text[start:number_end] + " " + c if c else None
        else:
            romaji = _map_kana(reading)
        if not romaji:
            continue
        idx = list(range(starts[start], ends[end] + 1))
        first, last = idx[0], idx[-1]
        tokens[first].surface, tokens[first].space = romaji, tokens[last].space
        for j in idx[1:]:
            tokens[j].surface, tokens[j].space = "", False


def _vowel_runs(seg):

    if not _VOWEL_RUN.search(seg):
        return seg
    words = v1.katsu.tagger(seg)
    if "".join(w.white_space + w.surface for w in words) != seg:
        return seg
    owner, pos = [None] * len(seg), 0
    for w in words:
        pos += len(w.white_space)
        for i in range(pos, pos + len(w.surface)):
            owner[i] = w.surface
        pos += len(w.surface)


    def sub(m):
        v = m.group(1)

        same = lambda tok: tok is not None and all(_VOWEL_OF.get(_hira(c)) == _hira(v) for c in tok)
        if "ァ" <= v <= "ヶ" or all(same(owner[i]) for i in range(m.start(), m.end())):
            prev = seg[m.start() - 1:m.start()]
            if re.fullmatch(_STRETCH_RE, prev):
                return _stretch(len(m.group()))
            if prev and _VOWEL_OF.get(_hira(prev)) == _hira(v):
                return "ー" + _stretch(len(m.group()) - 1)
            return v * 2 + _stretch(len(m.group()) - 2)
        return m.group(0)


    return _VOWEL_RUN.sub(sub, seg)


def _mark_stretches(seg):

    seg = re.sub(r"[〜～~]+", lambda m: _stretch(len(m.group())), seg)
    seg = re.sub(r"[ーｰ]{2,}", lambda m: "ー" + _stretch(len(m.group()) - 1), seg)
    seg = _vowel_runs(seg)

    seg = re.sub(r"([AIUEOＡＩＵＥＯaiueoａｉｕｅｏ])\1{2,}", lambda m: m.group(1) * 2 + _stretch(len(m.group()) - 2), seg)
    seg = re.sub(r"(.)([ぁぃぅぇぉァィゥェォ])(\2*)(ー*)(?=(.?))", _small_vowel, seg)

    return seg


def process_japanese_text_v2(ml, numbers="digits"):
    
    for a, b in zip("ぁぃぅぇぉ", "あいうえお"):
        ml = ml.replace(a, b)
    output = _romaji_v2(apply_transformations(alphabetreading(ml)), numbers).lower()


    output = output.replace("n'", _PROTECT["n'"]).replace("'", "")
    for k in ("fi", "di", "si"):
        output = output.replace(k, _PROTECT[k])

    output = output.replace("j", "dʑ").replace("tt", "ʔt").replace("t t", "ʔt").replace(" ʔt", "ʔt").replace("ssh", "ɕɕ")
    output = Roma2IPA(output)
    output = hira2ipa(output)
    output = replace_chars_2(output)
    output = replace_repeated_chars(replace_tashdid_2(output))
    output = nasal_mapper(output)

    for k in ("fi", "di", "si"):
        output = output.replace(_PROTECT[k], k)

    output = output.replace(" ɴ", "ɴ").replace("ctɕ", "ʔtɕ")
    if " neɽitai " in output:
        output = output.replace(" neɽitai ", "naɽitai")
    if "harɯdʑisama" in output:
        output = output.replace("harɯdʑisama", "arɯdʑisama")
    if "ki ni ɕinai" in output:
        output = re.sub(r"(?<!\s)ki ni ɕinai", r" ki ni ɕinai", output)
    if "de aɽoɯ" in output:
        output = re.sub(r"(?<!\s)de aɽoɯ", r" de aɽoɯ", output)


    return output.lstrip()


def _japanese_segment(segment_text, text_start=True, numbers="digits"):
    return _japanese_part(segment_text, segment_start=text_start, numbers=numbers)


def _japanese_part(segment_text, segment_start=True, numbers="digits"):
    original_segment = segment_text


    segment_text = re.sub(r"箱庭(?:作り|造り|づくり)", "ハコニワヅクリ", segment_text)
    segment_text = segment_text.replace("大黒柱", "大黒ばしら")
    segment_text = _mark_stretches(_join_split_long_vowels(segment_text))
    segment_text = re.sub(r"[っッ](?=[あいうえおアイウエオ、。！？…\s\ue101-\ue1ff]|$)", "◆", segment_text)
    segment_text = segment_text.replace("○", "|")

    output = post_fix(process_japanese_text_v2(segment_text, numbers))
    output = output.replace(" ɴ", "ɴ").replace("y", "j").replace("ɯa", "wa")
    if "niʔpoɴ go" in output: output = output.replace("niʔpoɴ go", "nihoɴ go")
    if "oɴ naː" in output: output = output.replace("oɴ naː", "onnaː")

    output = _random_space_fix_v2(output)
    output = random_sym_fix(output)
    output = random_sym_fix_no_space(output)
    output = output.lstrip().replace("... ", "...").replace("|", "○")

    if segment_start and output.startswith("wa") and original_segment.startswith("は"):
        output = "ha" + output[2:]

    output = output.replace("◆", "ʔ").replace("っ", "ʔ").replace("ッ", "ʔ").replace("ʔ♡", "♡♡")
    output = re.sub(r"\s*(" + _STRETCH_RE + ")", lambda m: "—" * (ord(m.group(1)) - _STRETCH0), output)

    output = output.replace(_PROTECT["n'"], "ɴ")

    output = re.sub(r"(\s*)" + BLOCK + r"(\s*)", lambda m: " " if m.group(1) or m.group(2) else "", output)
    output = v1.merge_katakana_phonemes(output, original_segment)

    return output.strip()


def phonemize_v2(text, bracket_english=False, numbers="digits"):


    processed = []
    for k, (segment_text, is_english) in enumerate(v1.split_japanese_english(text)):
        if is_english or v1.is_english_text(segment_text):
            processed.append(f"[{segment_text.strip()}]" if bracket_english else segment_text.strip())
        else:

            processed.append(_japanese_segment(segment_text, text_start=(k == 0), numbers=numbers))
    result = re.sub(r"\s+", " ", " ".join(processed))


    result = re.sub(r"\s+,", ",", result)
    result = re.sub(r"\[\s+", "[", result)
    result = re.sub(r",\s*(?=\[)", ", ", result)


    return result.replace(",ɴ", ", ɴ").replace("itɕi taikaɴ", "iʔtaikaɴ")
