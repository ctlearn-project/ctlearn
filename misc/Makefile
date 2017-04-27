CC=g++
CFLAGS=-Wall -g
CFLAGS+=`pkg-config --cflags`
CFLAGS+=`root-config --cflags`
LDFLAGS=`pkg-config --libs`
LDFLAGS+=`root-config --libs`

plotImgPars:
	 ${CC} plotImgPars.C -o plotImgPars ${CFLAGS} ${LDFLAGS}

generateEventList:
	 ${CC} generateEventList.C -o generateEventList ${CFLAGS} ${LDFLAGS}

.PHONY: clean
clean:
	rm -f plotImgPars
	rm -f generateEventList

.PHONY: all
all: clean plotImgPars generateEventList 
