import os
import math
import matplotlib.pyplot as plt #add to plot histgram
from utils import (Readtext, 
                   process_document, 
                   get_documents, 
                   extract_vocab, 
                   TopUsedwords, 
                   Fronted_adverbs, #add marker 1
                   Regular_adverbs, #add marker 2
                   Irregular_pt, #add marker 3
                   Regular_pt, #add marker 4
                   Initial_adverbs, #add marker 5
                   Delayed_adverbs, #add marker 6
                   Not_regular, #add marker 7
                   Not_contraction, #add marker 8
                   Pronoun_that, #add marker 9
                   Abbreviation, #add marker 10
                   )

#add
RED = '\033[31m'  #value output red
BLUE = '\033[34m'  #value output blue
GREEN = '\033[32m'  #value output green
YELLOW = '\033[33m'  #value output yellow
END = '\033[0m'  

#add　to plot histogram, but this is not used in final project.
def hist(score):
    labels = ["A","B","C","D"]
    plt.bar(labels, score)
    plt.title("Valus for each")
    plt.show()


def count_docs_in_class(documents, c):
    count=0
    for values in documents.values():
        if values[0] == c:
            count+=1
    return count


def Alldocs_in_class(documents,c):
    words_in_class = {}
    for d, values in documents.items():
        if values[0] == c:
            words_in_class.update(values[2])
    return words_in_class


def Training(classes, documents):  # Calculate the features of each author from the trainingdata.
    vocabulary = extract_vocab(documents)
    
    conditional_probabilities = {}
    for t in vocabulary:
        conditional_probabilities[t] = {}
    
    V = {}  # add
    for i in range(0,10): #amounts of markers:10
        V[i] = {}
        
    priors = {}
    class_size = {}
    alpha = 0.1
    print("\n\n<<Calculate features in each author>>")
    for c in classes:
        priors[c] = count_docs_in_class(documents,c) / len(documents)
        class_size = count_docs_in_class(documents, c)
        print("In class",c,"we have",class_size,"document(s).")
        words_in_class = Alldocs_in_class(documents,c)
         
        # add(Calculate the total number of counts for each of the 10 markers for each author.) 
        adverb_front_total = sum([documents[3] for documents in documents.values() if documents[0] == c])  # total count of marker 1
        adverb_reg_total = sum([documents[4] for documents in documents.values() if documents[0] == c])  # total count of marker 2
        pt_irreg_total = sum([documents[5] for documents in documents.values() if documents[0] == c])  # total count of marker 3
        pt_reg_total = sum([documents[6] for documents in documents.values() if documents[0] == c])  # total count of marker 4
        ini_adv_total = sum([documents[7] for documents in documents.values() if documents[0] == c])  # total count of marker 5
        del_adv_total = sum([documents[8] for documents in documents.values() if documents[0] == c])  # total count of marker 6
        not_reg_total = sum([documents[9] for documents in documents.values() if documents[0] == c])  # total count of marker 7
        not_cont_total = sum([documents[10] for documents in documents.values() if documents[0] == c])  # total count of marker 8
        that_total = sum([documents[11] for documents in documents.values() if documents[0] == c])  # total count of marker 9
        abbre_total = sum([documents[12] for documents in documents.values() if documents[0] == c])  # total count of marker 10
         
        denominator = sum(words_in_class.values()) #Parameter(Total number of words in text)
        for t in vocabulary:
            if t in words_in_class:
                conditional_probabilities[t][c] = (words_in_class[t] + alpha) / (denominator * (1 + alpha))  # unit5
            else:
                conditional_probabilities[t][c] = (0 + alpha) / (denominator * (1 + alpha))
            
        #add(Calculate probabilities at 10 authorship markers)
        V[0][c] = (adverb_front_total + alpha) / (denominator * (1 + alpha))  # probability of marker 1
        V[1][c] = (adverb_reg_total + alpha) / (denominator * (1 + alpha))  # probability of marker 2
        V[2][c] = (pt_irreg_total + alpha) / (denominator * (1 + alpha))  # probability of marker 3
        V[3][c] = (pt_reg_total + alpha) / (denominator * (1 + alpha))  # probability of marker 4
        V[4][c] = (ini_adv_total + alpha) / (denominator * (1 + alpha))  # probability of marker 5
        V[5][c] = (del_adv_total + alpha) / (denominator * (1 + alpha))  # probability of marker 6
        V[6][c] = (not_reg_total + alpha) / (denominator * (1 + alpha))  # probability of marker 7
        V[7][c] = (not_cont_total + alpha) / (denominator * (1 + alpha))  # probability of marker 8
        V[8][c] = (that_total + alpha) / (denominator * (1 + alpha))  # probability of marker 9
        V[9][c] = (abbre_total + alpha) / (denominator * (1 + alpha))  # probability of marker 10
                                
    return vocabulary, priors, conditional_probabilities, V


def Test(classes, priors, conditional_probabilities, test_document, V):  # Calculate score(probability) of each author from the testdata using the values of features obtained in Training().
    scores = {}
    author, doc_length, words = process_document(test_document) 
       
    text = Readtext(test_document)  # add(import the textfile)
    # add (count the value of each markers)
    adverb_front_count = Fronted_adverbs(text) # count the value of fronted adverbs of manner in Test data.
    adverb_reg_count = Regular_adverbs(text) # count the value of regular adverbs of manner in Test data.
    pt_irreg_count = Irregular_pt(text) # count the value of irregular past tenses in Test data.
    pt_reg_count = Regular_pt(text) # count the value of regular past tenses in Test data.
    initial_adv_count = Initial_adverbs(text) # count the value of sentence-initial transitional adverbs in Test data.
    delayed_adv_count = Delayed_adverbs(text) # count the value of delayed transitional adverbs in Test data.
    not_reg_count = Not_regular(text)  # count the value of regular "not" in Test data.
    not_cont_count = Not_contraction(text)  # count the value of contraction "not" in Test data.
    pronoun_count = Pronoun_that(text)  # count the value of pronoun "that" in Test data.
    abbreviation_count = Abbreviation(text)  # count the value of abbreviation words in Test data.
        
    for c in classes: # Loop for each author
        scores[c] = math.log(priors[c])
        
        for t in words:
            if t in conditional_probabilities:
                for i in range(words[t]):
                    scores[c] += math.log(conditional_probabilities[t][c])
        #add (Calculate the score for each authorship marker in the test file and add it to "score").
        scores[c] += math.log(V[0][c]) * adverb_front_count  # add score of marker 1 to "score"
        scores[c] += math.log(V[1][c]) * adverb_reg_count  # add score of marker 2 to "score" 
        scores[c] += math.log(V[2][c]) * pt_irreg_count  # add score of marker 3 to "score"
        scores[c] += math.log(V[3][c]) * pt_reg_count  # add score of marker 4 to "score"
        scores[c] += math.log(V[4][c]) * initial_adv_count  # add score of marker 5 to "score"
        scores[c] += math.log(V[5][c]) * delayed_adv_count  # add score of marker 6 to "score"
        scores[c] += math.log(V[6][c]) * not_reg_count  # add score of marker 7 to "score"
        scores[c] += math.log(V[7][c]) * not_cont_count  # add score of marker 8 to "score"
        scores[c] += math.log(V[8][c]) * pronoun_count  # add score of marker 9 to "score"
        scores[c] += math.log(V[9][c]) * abbreviation_count  # add score of marker 10 to "score"
        
    f = os.path.basename(test_document)
    print(GREEN+"Scores(probability) in descending order:"+END,f)
    print("--------------------------------------------------------")
    i = 0
    for author in sorted(scores, key=scores.get, reverse=True):
        i += 1
        print(i,". score(probability) of Author["+author+"]:",scores[author])
    print("--------------------------------------------------------")
    # add(Outputs the likely author of the test file)
    Author = max(scores,key=scores.get)
    print(YELLOW+"Comment:"+END+" Therefore, the author of <" + f + "> is most likely " +Author+ ".")  
    print("--------------------------------------------------------\n")
    # plot the histgram
    #hist(scores)    


#main関数

# import the textfiles in testdata.
files = [os.path.join('./dataset/testdata/', f) for f in sorted(os.listdir('./dataset/testdata/')) if os.path.isfile(os.path.join('./dataset/testdata/', f))]

classes = ["A","B","C","D"] # Author's name of the text in the training data
                            # The author's name is divided into A(Atherton), B(Rovert), C(Cornelia), and D(Lawrence) for clarity.
documents = get_documents() # utils

vocabulary, priors, conditional_probabilities, V = Training(classes, documents)

for author in classes:
    print("\n-------------------")
    print(RED+"Best features:"+END,author)
    print("-------------------")
    TopUsedwords(conditional_probabilities, author, 5)
    # add(Output features of added authorship markers in text files)
    print("\n----------------------------")
    print(BLUE+"Each authorship marker:"+END,author)
    print("----------------------------")
    print("[Fronted adverbs]:",V[0][author])
    print("[Rregular adverbs]:",V[1][author])
    print("[Irregular past tenses]:",V[2][author])
    print("[Regular past tenses]:",V[3][author])
    print("[Initial adverbs]:",V[4][author])
    print("[Delayed adverbs]:",V[5][author])
    print("[Not regular]:",V[6][author])
    print("[Not contraction]:",V[7][author])
    print("[Pronoun that]:",V[8][author])
    print("[Abbreviation]:",V[9][author])

# add(Loop to display the score for each textfile in the testdata.)
print("\n\n<<RESULTS>>\n")       
i = 0       
for f in files:
    if i != 0: 
       Test(classes, priors, conditional_probabilities, f, V)
    i += 1