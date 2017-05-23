CC=g++
CFLAGS= -Wall -g
CFLAGS+=`pkg-config --cflags`
CFLAGS+=`root-config --cflags`
LDFLAGS=`pkg-config --libs`
LDFLAGS+=`root-config --libs` -lTreePlayer 

plotImgPars:
	 ${CC} plotImgPars.C -o plotImgPars ${CFLAGS} ${LDFLAGS}

generateCutLists:
	 ${CC} generateCutLists.C -o generateCutLists ${CFLAGS} ${LDFLAGS}

generateBinLists:
	 ${CC} generateBinLists.C -o generateBinLists ${CFLAGS} ${LDFLAGS}

calculateAccuracy:
	 ${CC} calculateAccuracy.C -o calculateAccuracy ${CFLAGS} ${LDFLAGS}

plotBDT:
	 ${CC} plotBDT.C -o plotBDT ${CFLAGS} ${LDFLAGS}

.PHONY: clean
clean:
	rm -f plotImgPars
	rm -f generateCutLists
	rm -f generateBinLists
	rm -f calculateAccuracy
	rm -f plotBDT

.PHONY: all
all: clean plotImgPars generateCutLists generateBinLists calculateAccuracy plotBDT
