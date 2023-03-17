import stanza
import random
nlp = stanza.Pipeline(lang='en', use_gpu=False, processors='tokenize,pos') #,ner')

def getWordsPOSNER(text):

    sentwords = []
    postags = []
    nertags = []
    tagged = nlp (text)
    sentences = []

    for sentence in tagged.sentences:
        sent=""
        for token in sentence.tokens:

            word = token.words[0]
            sent +=" "+word.text
            sentwords.append (word.text)
            postags.append (word.upos)
            nertags.append (token.ner)

        sentences.append(sent.strip())

    return sentwords, postags, nertags, sentences


def getNPs(words, postags):
     kp_tags=[]
     
     for wx, word in enumerate(words):
         if (postags[wx]=="NOUN" or postags[wx]=="PROPN" or postags[wx]=="ADJ"):
             if wx==0 or (postags[wx-1]!="NOUN" \
                          and postags[wx-1]!="PROPN"
                          and postags[wx-1]!="ADJ"
                          and postags[wx-1]!="ADP"
                          and postags[wx-1]!="CCONJ") :
                kp_tags.append('B-KP')
             else:
                kp_tags.append('I-KP')
         elif (postags[wx]=="CCONJ" or postags[wx]=="ADP" ) and (wx<len(words)-1) and \
            (postags[wx+1]=="NOUN" or \
             postags[wx+1]=="PROPN"):
            kp_tags.append('I-KP')
         else:
            kp_tags.append('O')

     kpzones=[]
     covered=-1
     for wx, word in enumerate(words):
        
         if covered!=-1 and wx<covered:
             continue
        
        
        
         if kp_tags[wx].startswith("B"):
             temp = word
           
             for wx2 in range(wx+1, len(words)):
                 if kp_tags[wx2]!="O":
                     temp+=" "+words[wx2]
                 else:
                     break
            
             temp = temp.strip()
             if len(temp.split())==1:
                continue
             if temp not in kpzones:
                 kpzones.append(temp)
             covered = wx + len(temp.split())
            #print ("\n"+temp)

     return kpzones


def getNEZones(words, nertags):
    nerzones=[]

    covered=-1
    for wx, word in enumerate(words):

        if covered!=-1 and wx<covered:
            continue



        if nertags[wx].startswith("B") or nertags[wx].startswith("S"):
            temp = word

            for wx2 in range(wx+1, len(words)):
                if nertags[wx2]!="O":
                    temp+=" "+words[wx2]
                else:
                    break

            nerzones.append(temp.strip())
            covered = wx + len(temp.strip().split())
            #print ("\n"+temp)

    return nerzones

def removeSeen(seenkp, text):

    kws=[]
    for kp in seenkp:
        kws.extend(kp.lower().split())

    words = text.split()
    newtext=""
    for word in words:
        if word.lower().replace(",","").replace(".","").strip() not in kws:
            newtext +=" "+word

    return newtext


from nltk.corpus import stopwords
stoplist=stopwords.words('english')
#print (len(stoplist))
def removeSeen2(seenkp, tlines, mi):

    kws=[]

    for kp in seenkp:
        kpw=kp.split()
        seq=""
        for w in kpw:
            if w.lower() not in stoplist:
                seq+=" "+w
        if seq.strip()!="":
            kws.append(seq.lower())

    newtext=""
    removed=False
    newtl=[]
    for tline in tlines:
        tlw = tline.lower().split()
        tln =""
        covered=False
        for w in tlw:
            if w.lower() not in stoplist:
                tln+=" "+w
        for kw in kws:
            if tln in kw or kw in tln:
                if len(tln.replace(kw,"").split())<=3:
                    covered=True
                    break
        if not covered:
            newtext +=" "+tline
            newtl.append(tline)
        else:
            #print ("Removing covered line "+tline)
            removed=True

    if not removed and mi>=0:
        newtext=""
        newtl=[]
        for tx in range(len(tlines)):
            if tx!=mi:
                newtext +=" "+tlines[tx]
                newtl.append(tlines[tx])

    return newtext.strip(), newtl

def getNPsAndSentences(text):
    sentwords, postags, nertags, sentences  = getWordsPOSNER(text)
    return getNPs(sentwords, postags), sentences

if __name__=="__main__":
    text="Patient is hard of hearing. She also has vision problems. Denies headache syndrome. Presently, denies chest pain or shortness of breath. She denies abdominal pain. Presently, she has left hip pain and left shoulder pain. No urinary frequency or dysuria. No skin lesions. She does have swelling to both lower extremities for the last several weeks. She denies endocrinopathies. Psychiatric issues include chronic depression."


    kpzones, sl = getNPsAndSentences(text)
    print (kpzones)
    print (sl)
    print (kpzones[3])
    print (removeSeen2([kpzones[3]], sl, -1))


