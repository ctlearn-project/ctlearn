CC=g++
CFLAGS= -Wall -g
CFLAGS+=`pkg-config --cflags`
CFLAGS+=`root-config --cflags`
LDFLAGS=`pkg-config --libs`
LDFLAGS+=`root-config --libs` -lTreePlayer 

plotImgPars:
	 ${CC} plotImgPars.C -o plotImgPars ${CFLAGS} ${LDFLAGS}

calculateAccuracy:
	 ${CC} calculateAccuracy.C -o calculateAccuracy ${CFLAGS} ${LDFLAGS}

getBDTscores:
	 ${CC} getBDTscores.C -o getBDTscores ${CFLAGS} ${LDFLAGS}

.PHONY: clean
clean:
	rm -f plotImgPars
	rm -f calculateAccuracy
	rm -f getBDTscores

.PHONY: all
all: clean plotImgPars calculateAccuracy getBDTscores
