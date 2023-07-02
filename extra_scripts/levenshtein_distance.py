import numpy as np



"""
    for strings
"""
def levenshtein(sen1, sen2):
    if len(sen1) < len(sen2):
        return levenshtein(sen2, sen1)

    # len(sen1) >= len(sen2)
    if len(sen2) == 0:
        return len(sen1)

    previous_row = range(len(sen2) + 1)
    for i, c1 in enumerate(sen1):
        current_row = [i + 1]
        for j, c2 in enumerate(sen2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


print( levenshtein('Sienkiewicza', 'Mickiewicz'), '\n' )



"""
    for sequences (lists)
"""
def levenshteinDistanceDP( sentence_1, sentence_2 ):
    m, n = len(sentence_1), len(sentence_2)
    dists = np.zeros( (m + 1, n + 1), dtype=np.int16 ) # distances

    for t1 in range(m + 1):
        dists[t1][0] = t1

    for t2 in range(n + 1):
        dists[0][t2] = t2
    
    for t1 in range(1, m + 1):
        for t2 in range(1, n + 1):
            inser = dists[t1][t2-1] + 1
            delet = dists[t1-1][t2] + 1
            subst = dists[t1-1][t2-1] + (sentence_1[t1-1] != sentence_2[t2-1])
            dists[t1][t2] = min(inser, delet, subst)

    printDistances( dists, m, n )
    
    return dists[m][n]


def printDistances(dists, sentence_1_len, sentence_2_len):
    for t1 in range(sentence_1_len + 1):
        for t2 in range(sentence_2_len + 1):
            print( dists[t1][t2], end=" ")
        print()



sen_1 = ['chłopiec', 'mówi', 'tylko', 'że', 'miał', 'siedem', 'złotych']
sen_2 = ['Tucydydes', 'mówi', 'tylko', 'że', 'miał', 'siedem', 'okrętów']
sen_3 = ['ateński', 'dowódca', 'powiada', 'iż', 'dysponował', 'zbyt', 'małą', 'liczbą', 'statków']
sen_4 = ['the', 'genome', 'of', 'the', 'fungal', 'pathogen', 'that', 'causes', 'sudden', 'oak', 'death', 'has', 'been', 'sequenced', 'by', 'US', 'scientists']
sen_5 = ['researchers', 'announced', 'Thursday', 'they', 've', 'completed', 'the', 'genetic', 'blueprint', 'of', 'the', 'blight', 'causing', 'culprit', 'responsible', 'for', 'sudden', 'oak', 'death']
sen_6 = ['researchers', 'announced', 'Thursday', 'they', 've', 'completed', 'the', 'sketch', 'of', 'the', 'blight', 'causing', 'culprit', 'responsible', 'for', 'sudden', 'people', 'death']

distances = levenshteinDistanceDP(sen_5, sen_6)

print(distances)
print(len(sen_5))


