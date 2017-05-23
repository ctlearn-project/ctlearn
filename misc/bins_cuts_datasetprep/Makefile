CC=g++
CFLAGS= -Wall -g
CFLAGS+=`pkg-config --cflags`
CFLAGS+=`root-config --cflags`
LDFLAGS=`pkg-config --libs`
LDFLAGS+=`root-config --libs` -lTreePlayer 

generateCutLists:
	 ${CC} generateCutLists.C -o generateCutLists ${CFLAGS} ${LDFLAGS}

generateBinLists:
	 ${CC} generateBinLists.C -o generateBinLists ${CFLAGS} ${LDFLAGS}

.PHONY: clean
clean:
	rm -f generateCutLists
	rm -f generateBinLists

.PHONY: all
all: clean generateCutLists generateBinLists 
