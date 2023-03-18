from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as stutil
import SectionParser
from SectionParser import extractSections
from NPExtractor import getNPsAndSentences, removeSeen2
import numpy as np
import nltk
import random
import os
from util import get_device_string
import string
from DactPriors import loadDActProbs

#####
device=get_device_string()
from nltk.corpus import stopwords
stoplist=stopwords.words('english')
print ("Device "+device)
print (len(stoplist))


#######
secprobsf="resources/sec_dact_probs.txt"
dpranges, dpagg, ppranges, ppagg = loadDActProbs(secprobsf)
print (len(dpranges))
print (ppranges.keys())
def PBgetNextAct(speaker, secname):


    if speaker=="Doctor":
        if secname in dpranges:
            aggp = dpagg[secname]
            ch = random.randint(0,aggp)
            for dact in dpranges[secname]:
                (s, e) = dpranges[secname][dact]
                if ch>=s and ch<=e:
                    return dact +" [SEP] "+secname
        else:
            return "INFORM [SEP] "+secname
    else:
        if secname in ppranges:
            aggp = ppagg[secname]
            ch = random.randint(0,aggp)
            for dact in ppranges[secname]:
                (s, e) = ppranges[secname][dact]
                if ch>=s and ch<=e:
                    return dact +" [SEP] "+secname
        else:
            return "INFORM [SEP] "+secname


        

############


##
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2', device=device)
def getClosest(segments, sentences):

    #print ("DEBUG "+str(len(sentences)))
    if len(segments)==0 or len(sentences)==0:
        return -1, 0
    if len(sentences)>10:
        sentences=sentences[len(sentences)-10:]

    #print ("Here: "+str(len(sentences)))
    maxn=-1
    maxval=0
    for secn, segment in enumerate(segments):
        if len(segments[secn])==0:
            continue
        #print ("DEBUG "+str(secn)+" "+segments[secn])
        if len(segments[secn].strip().split())<3:
            continue
        embeddings = model.encode([segments[secn], sentences])
        dp = np.inner(embeddings[0], embeddings[1])
        den = np.sqrt(np.inner(embeddings[0], embeddings[0]) * np.inner(embeddings[1], embeddings[1]))

        if (dp/den) > maxval:
            maxval=(dp/den)
            maxn = secn

    if maxval>0.1:
        return maxn, maxval
    else:
        return -1, 0

#####################
def getClosest2(sentences, line):

    maxn=-1
    maxval=0
    for sx, sentence in enumerate(sentences):
        embeddings = model.encode([sentence.strip(), line.strip()])
        dp = np.inner(embeddings[0], embeddings[1])
        den = np.sqrt(np.inner(embeddings[0], embeddings[0]) * np.inner(embeddings[1], embeddings[1]))

        if (dp/den) > maxval:
            maxval=(dp/den)
            maxn = sx

    return maxn, maxval


def removeDuplicates(lines):

    
    covered=[]
    for lx in range(0, len(lines)):
        covered.append(False)

    for lx in range(0, len(lines)):

        if lx<2:
            continue

        if covered[lx]:
            continue

        lwords = lines[lx].split(":")[1].split()
        if len(lwords)<2:
            continue
        cnt=0
        for w in lwords:
            temp = w.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            if temp not in stoplist:
                cnt+=1

        if cnt>0:

            if lines[lx] in lines[0:lx-1]:
                print ("line seen exactly before: "+lines[lx])
                covered[lx]=True
                if lx+1 < len(lines):
                    covered[lx+1]=True
            else:
                ci, cv = getClosest2(lines[0:lx-1], lines[lx])
                if cv>0.9:
                    covered[lx]=True
                    if lx+1 < len(lines):
                        covered[lx+1]=True
        

    newlines=[]
    for lx in range(0, len(lines)):
        if not covered[lx]:
            newlines.append(lines[lx])

    return newlines


#####################

def cleanNP(nps):
    newnps=[]
    for kw in nps:
        kww=kw.lower().split()
        if len(kww)==1 and len(kww[0])<=3:
            continue
        found=False
        for key in SectionParser.secmap:
            if key.lower() in kww:
                found=True
                break

        if "patient" in kww or "medical" in kww:
            found=True

        if not found:
            newnps.append(kw)

    return newnps
        

#####################
gen_tokenizerdir="Elfsong/t5doctalk"
gen_dialogmodeldir="Elfsong/t5doctalk"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_tokenizerdir)
gen_model = T5ForConditionalGeneration.from_pretrained(gen_dialogmodeldir)
gen_model.to(device)

def getNextUtterance(input_string, **generator_args):
    input_ids = gen_tokenizer.encode(input_string, return_tensors="pt").to(device)
    res = gen_model.generate(input_ids, **generator_args)
    temp = gen_tokenizer.batch_decode(res, skip_special_tokens=True)[0]
    return temp.replace("</s>","").replace("<pad>","")


######
datokenizer = AutoTokenizer.from_pretrained("t5-large")
ddialog_model_dir="Elfsong/t5dact"
dialog_act_model = T5ForConditionalGeneration.from_pretrained(ddialog_model_dir).to(device)
act_types=["COMMISSIVE", "DIRECTIVE", "INFORM", "QUESTION"]

def parse_act_type(input_string,**generator_args):

    input_ids = datokenizer.encode(input_string, return_tensors="pt").to(device)
    res = dialog_act_model.generate(input_ids, **generator_args)
    op_string = datokenizer.batch_decode(res, skip_special_tokens=True)[0]

    for at in act_types:
        if at in op_string:
            return at

    return "INFORM"


###########

qa_model_name = "deepset/roberta-base-squad2"
if ":" in device:
    qa_nlp = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name, device=int(device.split(":")[1]))
else:
    qa_nlp = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name)

def runQA(question, plines):

    if question.strip()=="":
        return -1, ""

    passage = (' '.join(plines)).strip()
    QA_input = { 'question':question,
                'context':passage}
    result = qa_nlp(QA_input)
    return getCS(result, plines)

def getCS(result, plines):

    if result is None:
        return -1, ""
   # print ("Inside verifyAnswer for question="+question+", result = ")
   # print (result)
    cs = ""
    csi = -1
    si=[]
    ei=[]
    begin = 0
    for pline in plines:
        si.append(begin)
        ei.append(begin+len(pline.strip()))
        begin += len(pline.strip())+1


    if 'start' in result and 'end' in result:
        asi = result['start']
        aei = result['end']
        answer = result['answer']

        for px, pline in enumerate(plines):
            if answer in pline:
                if (abs(asi-si[px])<=5 or asi>=si[px]) and (abs(aei-ei[px])<=5 or aei<=ei[px]):
                    cs = pline.strip()
                    csi = px
                    break

    return csi, cs

##########
if ":" in device:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=int(device.split(":")[1]))
else:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


candidate_labels=[
        'GREETING',
        'AGREEMENT',
        'DENIAL',
        'ALLERGY',
        'ASSESSMENT', 'DIAGNOSIS','IMPRESSION','CONDITION',
        'PLAN','EDUCATION', 'COURSE', 'TREATMENT','INSTRUCTION',
        'COMPLAINT','ILLNESS',
        'HISTORY', 'FAMILY','PAST','FAMILY', 'SOCIAL', 'GYNIC','SURGICAL',
        'EXAMINATION','MEASUREMENT','PHYSICAL','VITALS',
        'IMAGING','IMMUNIZATIONS', 'LABS', 'PROCEDURES','RESULTS',
        'REVIEW', 'BODY', 'SYSTEMS','PHYSICAL',
        'MEDICATION',
        ]


def getZSLabel(seq):

    l = classifier(seq.strip(), candidate_labels)
    ops = l["labels"][0].strip()
    return ops

def bucketize(percent):
    if percent < 10:
        return "TEN"
    elif percent>=10 and percent<20:
        return "TWENTY"
    elif percent>=20 and percent<30:
        return "THIRTY"
    elif percent>=30 and percent<40:
        return "FORTY"
    elif percent>=40 and percent<50:
        return "FIFTY"
    elif percent>=50 and percent<60:
        return "SIXTY"
    elif percent>=60 and percent<70:
        return "SEVENTY"
    elif percent>=70 and percent<80:
        return "EIGHTY"
    elif percent>=80 and percent<90:
        return "NINTY"
    else:
        return "HUNDRED"



#####################



def getConversationFromReport(lines):
    
    sec2txt, orderedsecs = extractSections(lines)
    
    pu="START"
    puzsl=""
    dact=""
    nextspeaker=""
    
    tlines=[]
    nspeakers=[]
    seenutterances=[]

   # print (orderedsecs)

    print (sec2txt)
    seensec={}
    for secx, secnamep in enumerate(orderedsecs):
 #       if secx==1:
 #           break
        (secname, _) = secnamep
        if secname in seensec:
            continue

        if secname not in sec2txt: ##due to our filtering, some may go missing
            continue

        seensec[secname]=""
        print ("\nProcessing "+secname)

        
        orgsummary = sec2txt[secname].replace(secname, "").strip()
        allnps, sumlines = getNPsAndSentences(orgsummary)

        nps = cleanNP(allnps)
        covered_nps=[]
        summary = orgsummary.strip()
        print ("List of KP\n"+str(nps))
        nturns = max(len(nps)*2, len(sumlines)*2)
        print ("nturns="+str(nturns))
        extendonce=False
        for sx in range(0, nturns):

            if len(covered_nps)>0 and puzsl not in ['GREETING','AGREEMENT','DENIAL']:
                maxn, maxv = getClosest(sumlines, tlines)
                summary, sumlines = removeSeen2(covered_nps, sumlines, maxn)
                summary = summary.strip()
                
 
            if len(covered_nps) > 0.7*(len(nps)):
                break
            
            if len(covered_nps) < 0.5*(len(nps)) and sx==(nturns-4) and not extendonce:
                nturns = nturns*2
                extendonce=True

            #if len(sumlines)==0:
            #    break

            if len(nspeakers)==0 or nspeakers[-1]=="Patient":
                nextspeaker="Doctor"
            else:
                nextspeaker="Patient"

            nspeakers.append(nextspeaker)

            #print("========================")
            #print ("Round "+str(sx))

            npstr=""
            for np in nps:
                if np not in covered_nps:
                    npstr += np+" [SEP] "

            if npstr!="":
                npstr = npstr[0:len(npstr)-7]


            #print ("New summary: "+summary)
            
            pus=""
            pus2=""
            pu_person=""
            if len(tlines)>1:
                pus2 = tlines[-2].strip()

            if len(tlines)>0:
                pus = tlines[-1].strip()
                pu_person = pus.split(":")[0]
                pu = pus.split(":")[1]


            zslabel = puzsl
            sources = secname
            sources += " [SUMMARY] "+ summary.strip()
            sources += " [KWL] "+ npstr
            si=-1       
            acs=""
            if zslabel=="GREETING" and dact!="QUESTION":
                si=-1
            elif dact=="QUESTION" and zslabel!="GREETING":
                if len(sumlines)==0:
                    _, sumlines2 = getNPsAndSentences(orgsummary)
                else:
                    sumlines2=[]
                    sumlines2.extend(sumlines)

                if pu.strip()!="" and len(sumlines2)>0:
      #                  print ("pu for qa "+pu)
                    si, acs = runQA(pu, sumlines2)
             #       if si!=-1:
             #           print ("Found answer in: "+acs)

            if si!=-1 and acs!="":
                sources += " [ACS] "+acs

            if pus!="" and pus2=="":
                sources += " [PREVU] "+pus+" [PUDACT] "+dact
            elif pus!="" and pus2!="":
                sources += " [PREVU] "+pus2+" [SEP] "+pus+" [PUDACT] "+dact

            if zslabel!="":
                if zslabel not in ["QUESTION", "GREETING", "AGREEMENT", "DENIAL"]:
                    zslabel = "OTHER"

                sources += " [ZSL] "+zslabel

            #print ("BMI: "+sumlines[bmi])
            words = pu.split()
            kww=[]
            for word in words: 
                if len(word)<=3 or word.lower() in stoplist: 
                    continue
                else:
                    kww.append(word.lower())

            for kx, npkw in enumerate(nps):
                for w in kww:
                    if w in npkw.lower().split() and npkw not in covered_nps:
                        covered_nps.append(npkw)

            #print (covered_nps)

            if len(nps)!=0:
                percent = bucketize ((len(covered_nps)/len(nps))*100)
                pct2 = bucketize ((sx/len(nps)) * 100)
                sources += " [COVERED] PCT-"+str(percent)+" [TURNS] PCT-"+str(pct2)


            next_speaker_act = PBgetNextAct(nspeakers[-1], secname)
            #################

            ##For next round and for next prediction
            if "[SEP]" in next_speaker_act:
                dact = next_speaker_act.split("[SEP]")[0]  #using same variables for the next round 
                #puzsl = next_speaker_act.split("[SEP]")[1] #
            else:
                #Assign defaults based on secname?
                dact="INFORM"
                #puzsl="HISTORY"

            sources +=" [NDACT] "+dact+" "+nextspeaker.strip()+":"

            ###check for size here
            tempw = sources.strip().split()
            #print ("Length for u pred model "+str(len(tempw)))
            if len(tempw)>512:
                print ("Model input > 512, removing part of summary")
                tsum_prefix = sources.split("[SUMMARY]")[0]
                print (tsum_prefix)
                print (len(tempw)-1)

                tempw=tempw[len(tempw)-500:]
                temps = ' '.join(tempw)

                #print (temps)
                sources = (tsum_prefix+" [SUMMARY] "+temps).strip()
                #print ("New len of source="+str(len(sources.split())))



            ###

            predicted = getNextUtterance(sources.strip())
            puzsl = getZSLabel(predicted) ##for next round
            #print ("Next Speaker Utterance Prediction/ZSL "+predicted+"/"+puzsl)

            if predicted=="GSDASENDC":
                break

            t = nextspeaker.strip()+": "+predicted
            tlines.append(t)

    return tlines   


