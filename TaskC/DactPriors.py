import SectionParser
#######


def loadDActProbs(secprobsf):

    lines = open (secprobsf, "r").readlines()
    dprobs={}
    pprobs={}
    for lx in range(1, len(lines)):
        line = lines[lx]
        lp = line.strip().split()
        zsl = lp[0].strip()
        dact = lp[1].strip()
        dprob = int(lp[2])
        pprob = int(lp[3])

        if zsl in SectionParser.secmap:
            cansecn = SectionParser.secmap[zsl]

            if cansecn not in dprobs:
                dprobs[cansecn]={}
            if cansecn not in pprobs:
                pprobs[cansecn]={}


            if dact not in dprobs[cansecn]:
                dprobs[cansecn][dact]=0
            if dact not in pprobs[cansecn]:
                pprobs[cansecn][dact]=0

            dprobs[cansecn][dact] += dprob
            pprobs[cansecn][dact] += pprob

    dpranges={}
    dpagg={}
    for secname in dprobs:
        dpr={}
        dacts = dprobs[secname]
        aggp = 0
        start=0
        for dact in dacts:
            end = start+dprobs[secname][dact]
            dpr[dact]=(start, end)
            start = end
            aggp += dprobs[secname][dact]

        dpranges[secname]=dpr
        dpagg[secname]=aggp

    ppranges={}
    ppagg={}
    for secname in pprobs:
        ppr={}
        dacts = pprobs[secname]
        start=0
        aggp=0
        for dact in dacts:
            end = start+pprobs[secname][dact]
            ppr[dact]=(start, end)
            start = end
            aggp += pprobs[secname][dact]

        ppranges[secname]=ppr
        ppagg[secname]=aggp

    return dpranges, dpagg, ppranges, ppagg

if __name__=="__main__":
    secprobsf="resources/sec_dact_probs.txt"
    dpranges, dpagg, ppranges, ppagg = loadDActProbs(secprobsf)

    print (dpagg)
    #print (dpranges)
    print ("\n")
    for key in dpranges:
        print (key+" "+str(dpranges[key]))

    print ("\n\n")
    print (ppagg)
    print ("\n")
    for key in ppranges:
        print (key+" "+str(ppranges[key]))
