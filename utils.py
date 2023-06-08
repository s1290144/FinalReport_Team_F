import re
import os

def get_documents():
    documents = {}
    files = [os.path.join('./dataset/trainingdata/', f) for f in os.listdir('./dataset/trainingdata/') if os.path.isfile(os.path.join('./dataset/trainingdata/', f))]
    for f in files:
        author, doc_length, words = process_document(f)
                    
        text = Readtext(f)   # add(import the textfile) 
        adverb_fronted_count = Fronted_adverbs(text)  # add marker 1 (fronted adverbs of manner)
        adverb_regular_count = Regular_adverbs(text)  # add marker 2 (regular adverbs of manner)
        pt_regular_count = Regular_pt(text)  # add marker 3 (irregular past tenses)
        pt_irregular_count = Irregular_pt(text)  # add marker 4 (regular past tenses)
        initial_adverb_count = Initial_adverbs(text)  # add marker 5 (sentence-initial transitional adverbs)
        delayed_adverb_count = Delayed_adverbs(text)  # add marker 6 (delayed transitional adverbs)
        not_reg_count = Not_regular(text)   # add marker 7 (regular "not")
        not_cont_count = Not_contraction(text)   # add marker 8 (contraction "not")
        pronoun_count = Pronoun_that(text)   # add marker 9 (Pronoun "that")
        abbreviation_count = Abbreviation(text)   # add marker 10 (Abbreviation words)
         
        documents[f] = [author, doc_length, words, 
                        adverb_fronted_count, 
                        adverb_regular_count, 
                        pt_irregular_count, 
                        pt_regular_count,
                        initial_adverb_count,
                        delayed_adverb_count,
                        not_reg_count,
                        not_cont_count,
                        pronoun_count,
                        abbreviation_count,
                        ]
    return documents

def extract_vocab(documents):
    vocabulary = []
    for values in documents.values():
        vocabulary += list(values[2].keys())
    return vocabulary

def Readtext(textfile):  # add(import the textfile)
    with open(textfile, 'r' ,encoding='utf-8') as file:
        text = file.read()
    return text

def Fronted_adverbs(text):  # add marker 1 (fronted adverbs of manner)
    found = re.findall("ly ", text)  # Find all adverbs with ly before the verb in the text file.
    cnt = len(found)  # Count all adverbs with ly before the verb in the text file.
    return cnt

def Regular_adverbs(text):  # add marker 2 (regular adverbs of manner)
    found = re.findall("ly.", text)  # Find all adverbs with ly at the end of the sentence in the text file.
    cnt = len(found)  # Count all adverbs with ly at the end of the sentence in the text file.
    return cnt

def Irregular_pt(text):  # add marker 3 (irregular past tenses)
    PV = ["arose",
          "awoke",
          "bore",
          "beat",
          "became",
          "began",
          "bent",
          "bet",
          "bit",
          "blew",
          "broke",
          "brought",
          "built",
          "burst",
          "bought",
          "caught",
          "chose",
          "came",
          "cost",
          "cut",
          "dealt",
          "dug",
          "did",
          "drew",
          "drank",
          "drove",
          "ate",
          "fell",
          "felt",
          "fought",
          "found",
          "flew",
          "forgot",
          "forgave",
          "froze",
          "got",
          "gave",
          "went",
          "grew",
          "hung",
          "had",
          "heard",
          "hid",
          "hit",
          "held",
          "hurt",
          "kept",
          "knew",
          "laid",
          "led",
          "left",
          "lent",
          "let",
          "lay",
          "lit",
          "lost",
          "made",
          "meant",
          "met",
          "paid",
          "put",
          "quit",
          "read",
          "rode",
          "rang",
          "rose",
          "ran",
          "said",
          "saw",
          "sold",
          "sent",
          "set",
          "shook",
          "shone",
          "shot",
          "showed",
          "shut",
          "sang",
          "sank",
          "sat",
          "slept",
          "slid",
          "spoke",
          "sped",
          "spent",
          "spun",
          "spread",
          "stood",
          "stole",
          "stuck",
          "stung",
          "struck",
          "swore",
          "swept",
          "swam",
          "swung",
          "took",
          "taught",
          "tore",
          "told",
          "thought",
          "threw",
          "understood",
          "woke",
          "wore",
          "won",
          "withdrew",
          "wrote"] # list of irregular past tenses
    cnt = 0
    for pv in PV:
        found = re.findall(pv,text)  # Find all past tenses of irregular verbs in text files.
        cnt += len(found)  # Count all past tenses of irregular verbs in text files
    return cnt

def Regular_pt(text):  # add marker 4 (regular past tenses)
    found = re.findall("ed ",text)  # Find all past tenses of regular verbs in text files.
    cnt = len(found)  # Count all past tenses of regular verbs in text files
    return cnt

def Initial_adverbs(text):  # add marker 5 (sentence-initial transitional adverbs)
    IA = ["Besides",
          "Also",
          "Moreover",
          "Then",
          "Additionally",
          "Furthermore",
          "However",
          "Nevertheless",
          "Nonetheless",
          "Still",
          "Yet",
          "All the same",
          "On the other hand",
          "Meanwhile",
          "In contrast",
          "Or else",
          "Otherwise",
          "That is",
          "That is to say",
          "Namely",
          "Indeed",
          "For example",
          "For instance",
          "So",
          "Therefore",
          "Thus",
          "Consequently",
          "Accordingly",
          "Hence",
          "As a result",
          ] # list of sentence-initial transitional adverbs
    cnt = 0
    for ia in IA:
        found = re.findall(ia,text)  # Find all sentence-initial transitional adverbs in text files.
        cnt += len(found)  # Count all sentence-initial transitional adverbs in text files.
    return cnt

def Delayed_adverbs(text):  # add marker 6 (delayed transitional adverbs)
    IA = ["besides",
          "also",
          "moreover",
          "then",
          "additionally",
          "furthermore",
          "however",
          "nevertheless",
          "nonetheless",
          "still",
          "yet",
          "all the same",
          "on the other hand",
          "meanwhile",
          "in contrast",
          "or else",
          "otherwise",
          "that is",
          "that is to say",
          "namely",
          "indeed",
          "for example",
          "for instance",
          "so",
          "therefore",
          "thus",
          "consequently",
          "accordingly",
          "hence",
          "as a result",
          ] # list of delayed transitional adverbs
    cnt = 0
    for ia in IA:  # Find all delayed transitional adverbs in text files.
        found = re.findall(ia,text)  # Count all delayed transitional adverbs in text files.
        cnt += len(found)
    return cnt

def Not_regular(text):  # add marker 7 (regular "not")
    NOT = [" are not",
           " is not",
           " was not",
           " were not",
           " will not",
           " have not",
           " has not",
           " had not",
           " can not",
           " cannot ",
           ] # list of regular "not" ("am" is abbreviation because not is not changed to "n't")
    cnt = 0
    for ia in NOT:
        found = re.findall(ia,text)  # Find all regular "not" in text files.
        cnt += len(found)  # Count all regular "not" in text files.
    return cnt

def Not_contraction(text):  # add marker 8 (contraction "not")
    NOT = [" aren't",
           " isn't",
           " wasn't",
           " weren't",
           " won't",
           " haven't",
           " hasn't",
           " hadn't",
           " can't",
           ] # list of contraction "not"
    cnt = 0
    for ia in NOT:
        found = re.findall(ia,text)  # Find all contraction "not" in text files.
        cnt += len(found)  # Count all contraction "not" in text files.
    return cnt

def Pronoun_that(text):  # add marker 9 (Pronoun "that")
    found = re.findall(" that ",text)  # Find all pronoun "that" in text files.
    cnt = len(found)  # Count all pronoun "that" in text files.
    return cnt

def Abbreviation(text):  # add marker 10 (Abbreviation words)
    ABB = ["etc.",
           "e.g.",
           "i.e.",
           ]  # list of Addreviation words 
    cnt = 0
    for ia in ABB:
        found = re.findall(ia,text)  # Find all addreviation words in text files.
        cnt += len(found)  # Count all addreviation words in text files.
    return cnt

def process_document(textfile):
    words = {}
    doc_length = 0
    f = open(textfile,'r', encoding='utf-8')
    c = 0
    for l in f:
        l = l.rstrip('\n')
        if c < 1:
            author = l.replace("#Author: ", "")
        else:
            word_list = l.split()
            for i in range(len(word_list)):
                w = word_list[i]
                if w in words:
                    words[w] += 1
                else:
                    words[w] = 1
                doc_length += 1
        c += 1
    f.close()
    return author, doc_length, words


def TopUsedwords(conditional_probabilities,author,n):
    cps = {}
    for term,probs in conditional_probabilities.items():
        cps[term] = probs[author]    
    c = 0
    for term in sorted(cps, key=cps.get, reverse=True):
        if c < n:
            print(c,term,"score:",cps[term])
            c+=1
        else:
            break