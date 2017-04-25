CC=g++
CFLAGS=-Wall -g
CFLAGS+=`pkg-config opencv exiv2 --cflags`
CFLAGS+=`root-config --cflags`
LDFLAGS=`pkg-config opencv exiv2 --libs`
LDFLAGS+=`root-config --libs`

plotImgPars:
	 ${CC} plotImgPars.C -o plotImgPars ${CFLAGS} ${LDFLAGS}

generateImageList:
	 ${CC} generateImageList.C -o generateImageList ${CFLAGS} ${LDFLAGS}

.PHONY: clean
clean:
	rm -f plotImgPars
	rm -f generateImageList

.PHONY: all
all: clean plotImgPars generateImageList 
