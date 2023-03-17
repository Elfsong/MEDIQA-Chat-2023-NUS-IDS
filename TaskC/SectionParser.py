#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:30:29 2023

@author: sdas
"""


import string
import os
#######################
secmap={
        'ALLERGIES':'ALLERGY',
        'ALLERGY':'ALLERGY',
        ##
        'ASSESSMENT':'ASSESSMENT',
        'DIAGNOSIS':'ASSESSMENT',
        'IMPRESSION':'ASSESSMENT',
        'CONDITION':'ASSESSMENT',
        ##
        'PLAN':'PLAN',
        'EDUCATION':'PLAN',
        'COURSE':'PLAN',
        'TREATMENT':'PLAN','INSTRUCTION':'PLAN',
        'DISPOSITION':'PLAN',
        ##
        'COMPLAINT':'COMPLAINT','ILLNESS':'COMPLAINT', 'CC':'COMPLAINT',
        ##
        'FAM/SOCHX':'HISTORY',
        'GENHX':'HISTORY',
        'HX':'HISTORY',
        'PASTMEDICALHX':'HISTORY',
        'HISTORY':'HISTORY',
        'FAMILY':'HISTORY',
        'PAST':'HISTORY',
        'SOCIAL':'HISTORY',
        'GYNIC':'HISTORY',
        'SURGICAL':'HISTORY',
        ##
        'EXAMINATION':'EXAMINATION',
        'MEASUREMENT':'EXAMINATION',
        'PHYSICAL':'EXAMINATION',
        'VITALS':'EXAMINATION',
        'IMAGING':'EXAMINATION',
        'IMMUNIZATIONS':'EXAMINATION',
        'LABS':'EXAMINATION',
        'PROCEDURES':'EXAMINATION',
        'RESULTS':'EXAMINATION',
        ##
        'REVIEW':'REVIEW', 'ROS':'REVIEW', 'SYSTEMS':'REVIEW',
        ##
        'MEDICATION':'MEDICATION','MEDICATIONS':'MEDICATION'

        }

def getSecName(secname, orderedsecs):

    for key in secmap:
        if secname in key or key in secname:
            return secmap[key]

    print ("Did not find "+ secname+" in map")
    if len(orderedsecs)>0:
        (n, _) = orderedsecs[-1]
        print ("Returning previous: "+n)
        return n
    else:
        print ("Returning GENERAL")
        return "GENERAL"



def extractSections(lines):

    l2sec={}
    orderedsecs=[]
    secname=""
    for lx, line in enumerate(lines):
        if line.strip().isupper() and len(line.strip().split())<5:
            secname=getSecName(line, orderedsecs)
            if secname!="" and secname not in orderedsecs:
                orderedsecs.append((secname, lx))
                continue

        if secname!="":
            l2sec[lx]=secname
    
    print ("#lines "+str(len(lines))+" #l2sec "+str(len(l2sec)))
    sec2txt={}
    
    for lx, line in enumerate(lines):
        if lx not in l2sec:
            continue
        secn = l2sec[lx]

        temp = line.translate(str.maketrans('', '', string.punctuation)).strip()
        if len(temp.split())<2:
           # print ("Ignoring line in note "+line)
            continue

        if secn not in sec2txt:
            sec2txt[secn] = line.strip()
        else:
            sec2txt[secn] += " " +line.strip()
 
    newsec2txt={}
    for secn in sec2txt:
        if len(sec2txt[secn].split()) < 5:
            continue
        else:
            newsec2txt[secn]=sec2txt[secn]

    return newsec2txt, orderedsecs



