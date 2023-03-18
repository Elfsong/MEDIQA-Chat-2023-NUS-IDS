#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:41:44 2023

@author: sdas
"""

import csv
import sys
from ConvGenerator3 import getConversationFromReport, removeDuplicates

IDCOL="encounter_id"
SRCID="note"


def parseCSV(inpf):
    eid2note={}
    rkeys={}
    with open(inpf, "r") as f:
        data = csv.reader(f)
        for rx, row in enumerate(data):
            if rx==0:
                for cx, col in enumerate(row):
                    rkeys[col]=cx
                continue
            
            eid = row[rkeys[IDCOL]]
            src = row[rkeys[SRCID]]
            eid2note[eid] = src
            
            
        print (rkeys)
    
    return eid2note

if __name__=="__main__":
    
    if len(sys.argv)!=3:
        sys.exit("input/output CSV filepaths expected")

    inpf = sys.argv[1]
    outf = sys.argv[2]

    print ("Input file: "+inpf)
    print ("Output file: "+outf)

    eid2note = parseCSV(inpf)
    header=['TestID','SystemOutput']

    with open(outf, 'w', encoding='UTF8') as f:
        
        writer = csv.writer(f)
        writer.writerow(header)

        for eid in eid2note:
            if eid!="D2N136":
                continue
            print ()
            print ("Processing "+eid)
            note = eid2note[eid]
            conv = getConversationFromReport(note.split("\n"))
            print ("Lines in note "+str(len(note.split("\n"))))
            print ("Conversation length "+str(len(conv)))
            cleanconv = removeDuplicates(conv)
            print ("After Duplicate Removal "+str(len(cleanconv)))

            fout = open (eid+".tmp3","w")
            for cl in conv:
                fout.write(cl.strip()+"\n")
                fout.flush()
            fout.write("========================\n")
            for cl in cleanconv:
                fout.write(cl.strip()+"\n")
                fout.flush()
            fout.close()

            tempstr=""
            for cl in cleanconv:
                tempstr +="\n"+cl.strip()



            row = [eid, tempstr.strip()]
            writer.writerow(row)

    f.close()

    
