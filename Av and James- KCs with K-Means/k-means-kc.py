import matplotlib.pyplot as plt
from decimal import *
import numpy as np
from numpy import *
import xlrd
import pyexcel.ext.xlsx as pe
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
# from scipy.cluster.vq import vq, kmeans
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA

X = []
Y = []

Students = []
KCs = []
trainingRows = []
studentIDs = []

def read_data_from_file():
    print "HELLO"
    kc = []
    tempKc = []
    students = []
    global X, Y, Students,  studentIDs
    x = []
    y = []
    global KCs
    readFile = open('algebra_2005_2006_train.txt', 'r')
    sampleData = readFile.read().split('\n')
    count = 0
    f = open("kc1.txt", "w")
    f2 = open("Nokc.txt", "w")
    f3 = open("newKc.txt", "w")
    kcTest = []

    for data in sampleData:

        data = data.lower()
        if( data != "" ):
            trainingRows.append(data)
        seperateData = data.split('\t')
        # f.write(seperateData[17])
        # if count == 2:
        #     exit();
        # count += 1;

        # print seperateData[17]
        # print seperateData[1]

        if( len( seperateData ) >= 17 ):
            if (seperateData[1] not in studentIDs):
                studentIDs.append(seperateData[1])

            if( seperateData[17].find( "~~" ) ):
                tempKc = seperateData[17].split("~~")
                for val in tempKc:
                    # kcTest.append(val)
                    #val = val.replace('skillrule:', '') # Needed
                    if not (val.isspace()):
                        if( val not in kc ):
                            # val = val.strip()
                            kc.append(val)
                            f.write(val)
                            f.write("\n")
                    # if( val.startswith("[skillrule: ") ):
                    #     val = val.replace("[skillrule: ", "")
                    #     val = val[:-1]
                    #     val = val.split(";")
                    #     for tempVal in val:
                    #         if( tempVal not in kc ):
                    #             f.write(tempVal.lstrip(' '))
                    #             f.write("\n")
                    #             kc.append( tempVal )
            else:
                if seperateData[17] != "":
                    # kcTest.append(seperateData[17])
                    #seperateData[17] = seperateData[17].replace('skillrule', '')  # Needed
                    if (seperateData[17] not in kc):
                        seperateData[17] = seperateData[17].strip()
                        kc.append(seperateData[17])
                        f.write(seperateData[17])
                        f.write("\n")
                else:
                    continue

    print kc
    kc.remove('kc(default)')
    kc = filter(None, kc)
    # kc.remove('\n')
    for val in kc:
        f.write(val)
        f.write("\n")
    students = np.zeros((len(studentIDs), len(kc)))
    print "------------------- Shape"
    print students.shape

    studentIndex = 0

    for singleStudentID in studentIDs:

        studentRows = filter(lambda x: singleStudentID in x, trainingRows)
        for singleRow in studentRows:
            for singleKc in kc:
                # print "Single Row"
                # print singleRow
                # print "Single KC"
                # print singleKc
                # print "------------------------"
                if singleKc in singleRow:
                    seperateData = singleRow.split('\t')
                    if seperateData[17].find('~~'):
                        tempKcs = seperateData[17].split("~~")
                        # print "HI"
                        # for val in tempKcs:
                        #     print val
                        tempKcs.index(singleKc) # Retriving index from kc as it will be same for Opportinuty cost
                        tempOpportinuty = seperateData[18].split('~~')
                        if int(seperateData[13]) == 1 and int(tempOpportinuty[tempKcs.index(singleKc)]) != 0:
                            students[studentIndex][kc.index(singleKc)] += 1.0 * float(tempOpportinuty[tempKcs.index(singleKc)])
                            students[studentIndex][kc.index(singleKc)] = "{0:.2f}".format(students[studentIndex][kc.index(singleKc)])
        if studentRows in trainingRows:
            trainingRows.remove(studentRows)
        studentIndex += 1

    np.savetxt('output.txt', students)

    # Using bag of words, didn't work
    vectorizer = CountVectorizer()
    print vectorizer.fit_transform(kc).todense()
    print vectorizer.vocabulary_
    for v in vectorizer.vocabulary_:
        print v
        f3.write(v)
        f3.write("\n")


    # Elbow Method
    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(students)
        meandistortions.append(sum(np.min(cdist(students, kmeans.cluster_centers_, 'euclidean'), axis = 1)) / students.shape[0])

    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Average distortion')
    plt.title('Selecting k with the Elbow Method')
    plt.show()

    # Best fit for k = 3
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(students)


    # Dimensionality reduction
    pca = PCA(n_components=3)




    # f.write(kc)
    #     tempKC = seperateData[10].split('~~')
    #     for text in tempKC:
    #         print count
    #         # text = text.replace("\"", "")
    #         # text = text.replace("SkillRule: ", "")
    #         # text = text.replace("[", "")
    #         # text = text.replace("]", "")
    #         # text = text.split(';')
    #         count+=1
    #         if not text in KCs:
    #             # print type(text)
    #             KCs.append(text)
    #     #     # for tempText in text:
    #     #     #     if not text in KCs:
    #     #     #         print type(text)
    #     #     #         KCs.append(text)
    #
    #
    # print KCs
    # print len(KCs)
    f.close()
    exit()

read_data_from_file()