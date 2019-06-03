import re
import string
from functools import partial
from nltk.tokenize import word_tokenize as nltk_tokenize
from collections import OrderedDict
from pygtrie import StringTrie
import tldextract


def _replace(mentions, replace_word=' ', pattern=re.compile('')):
    mentions = [pattern.sub(replace_word, mention) for mention in mentions]
    return mentions


PATTERN_PHONENB = re.compile(
    r'(\(\+?0?84\))?(03|04|05|07|08|09|012|016|018|019)((\d(\s|\.|\,)*){8})')
SUB_PHONENB = r' PHONEPATT '

PATTERN_URL = re.compile(
    r'(((ftp|https?)\:\/\/)|(www\.))?[\d\w\.\-\_]+\.[\w]{2,6}(:[\d\w]+)?[\#\d\w\-\.\_\?\,\'\/\\\+\;\%\=\~\$\&]*(.html?)?')
SUB_URL = r' URLPATT '

PATTERN_EMAIL = re.compile(
    r'(^|\W)([^@\s]+@[a-zA-Z0-9\-][a-zA-Z0-9\-\.]{0,254})(\W|$)')
SUB_EMAIL = r' EMAILPATT '

PATTERN_NUMBER = re.compile(r'((\d+(\s|\.|\,|-){,2}\d*){2,})')
SUB_NUMBER = r' NUMPATT '

PATTERN_HTMLTAG = re.compile(r'<[^>]*>')

PATTERN_PUNCTATION = re.compile(r'([%s])' % re.escape(string.punctuation))

PATTERN_LINEBRK = re.compile(r'\t|\v|\f|(\s){2,}|\r\n|\r|\n')

PATTERN_NOT_PUNC_WSPACE_ALNUM = re.compile(r'[^%s\w\d]' % re.escape(
                    string.punctuation + string.whitespace), re.UNICODE)

PATTERN_FULLNAME = re.compile(r'([BCDĐGHKLMNPQRSTVXAÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌƠỚỜỞỠỢÔỐỒỔỖỘUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ][bcdđghklmnpqrstvxaáàảãạâấầẩẫậăắằẳẵặeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọơớờởỡợôốồổỗộuúùủũụưứừửữựyýỳỷỹỵ]+\s?){2,}', re.UNICODE)


def remove_html_tag(mentions):
    # Replace html tag
    mentions = _replace(mentions, pattern=PATTERN_HTMLTAG)
    return mentions


def remove_name(mentions):
    # Name (of human, place, etc.) is usually capitalized the first letter
    # Remove the word which has the first capital character
    mentions = _replace(mentions, pattern=PATTERN_FULLNAME)
    return mentions


def remove_line_break(mentions):
    # Remove line break
    mentions = _replace(mentions, pattern=PATTERN_LINEBRK)
    return mentions


def remove_special_chars(mentions):
    # Remove the special characters that are not listed in the keyboard
    mentions = _replace(mentions, pattern=PATTERN_NOT_PUNC_WSPACE_ALNUM)
    return mentions


def remove_oov(mentions, vocab_dict):
    # Remove the word that is not in the provided vocabulary dictionary
    mentions = [' '.join(word for word in mention.split() if word in vocab_dict) for mention in mentions]
    return mentions


def replace_phoneNB(mentions):
    # Replace phone number
    mentions = _replace(mentions, SUB_PHONENB, PATTERN_PHONENB)
    return mentions


def replace_email(mentions):
    # Replace email
    mentions = _replace(mentions, SUB_EMAIL, PATTERN_EMAIL)
    return mentions


def replace_all_number(mentions):
    # Replace all remained number
    mentions = _replace(mentions, SUB_NUMBER, PATTERN_NUMBER)
    return mentions


def _get_trie(*list_replacement_items):
    replacement_items_ordereddict = OrderedDict()

    for replacement_items in list_replacement_items:
        replacement = replacement_items[0]
        items = replacement_items[1]
        replacement_items_ordereddict[replacement] = items

    trie = StringTrie(separator=' ')

    for key in replacement_items_ordereddict.keys():
        trie.update(
            StringTrie.fromkeys(
                replacement_items_ordereddict[key],
                value=key,
                separator=' '))
    return trie


def _replace_entities_for_mention(mention, trie):
    tokens = tokenize_by_punc([mention])[0].split()
    del_indices = set()
    for token_index in range(len(tokens)):
        current_token = tokens[token_index]

        if token_index in del_indices:
            continue

        if trie.has_node(current_token):
            current_node = current_token
            look_forward_num = 1

            while True:
                if token_index + look_forward_num < len(tokens):
                    next_node = current_node + ' ' + \
                                tokens[token_index + look_forward_num]
                else:
                    next_node = None

                if next_node is None or not trie.has_node(next_node):
                    if trie.has_key(current_node):
                        tokens[token_index] = trie[current_node]
                        del_indices.update(
                            range(
                                token_index + 1,
                                token_index + look_forward_num))
                    break
                else:
                    current_node = next_node
                    look_forward_num += 1

    replaced_mention = ' '.join(token for index, token in enumerate(tokens) if index not in del_indices)
    return replaced_mention


def replace_entities_keywords(mentions, entities=[], keywords=[]):
    trie = _get_trie((' ENTITY ',entities),(' KEYWORD ',keywords))
    mentions = [_replace_entities_for_mention(mention, trie) for mention in mentions]
    return mentions


def lowercase(mentions):
    mentions = [mention.lower() for mention in mentions]
    return mentions


def keep_max_word_len(mentions, max_word_len=10):
    # only keep the word that no longer than the max length.
    mentions = [' '.join(word for word in mention.split() if len(word) <= max_word_len) for mention in mentions]
    return mentions


def tokenize_by_punc(mentions):
    mentions = [PATTERN_PUNCTATION.sub(r' \1 ', mention) for mention in mentions]
    return mentions


def nltk_tokenizer(mentions):
    # use nltk library
    mentions = [' '.join(nltk_tokenize(mention)) for mention in mentions]
    return mentions


def replace_url(mentions, keep_url_host=True):
    # Replace url
    if keep_url_host:
        mentions = _replace(mentions, partial(_get_domain_name), PATTERN_URL)
    else:
        mentions = _replace(mentions, SUB_URL, PATTERN_URL)
    return mentions


def _get_domain_name(match):
    return tldextract.extract(match.group(0)).domain