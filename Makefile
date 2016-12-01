MAJOR  ?= 0
MINOR  ?= 0
SUB    ?= 1
PATCH  ?= 1

SCRAPPIE_VERSION = $(MAJOR).$(MINOR).$(SUB)-$(PATCH)


CC     ?= gcc
LIBS    = -lblas -lhdf5 -lm
CFLAGS  = -Wall -Wno-unused-function -O3 -fopenmp -march=core2 -ffast-math -std=c99 -DUSE_SSE2 -DSCRAPPIE_VERSION=\"$(SCRAPPIE_VERSION)\"
OBJDIR  = obj
OBJECTS = read_events.o features.o util.o layers.o decode.o
SEDI    = sed -i
MD5SUM  = md5sum

ifeq ($(shell uname), Darwin)
        SEDI   = sed -i ""
        MD5SUM = md5 -r
endif

all: basecall

%.o: %.c
	$(CC) $(INC) -c -o $@ $< $(CFLAGS)

basecall.c: decode.h features.h layers.h read_events.h util.h lstm_model.h
decode.c: decode.h
decode.h: util.h
features.c: features.h
features.h: read_events.h util.h
layers.c: layers.h util.h
layers.h: util.h
read_events.c: read_events.h
util.c: util.h
util.h: sse_mathfun.h





basecall: basecall.o $(OBJECTS) lstm_model.h
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -f *.o basecall

deps:
	grep ^Depends DEBIAN/control | cut -d : -f 2 | sed 's/,/ /g' | sed 's/([^)]*)//g' | xargs apt-get install -y --force-yes
	grep ^Build-Depends DEBIAN/control | cut -d : -f 2 | sed 's/,/ /g' | sed 's/([^)]*)//g' | xargs apt-get install -y --force-yes
	grep ^Build-Recommends DEBIAN/control | cut -d : -f 2 | sed 's/,/ /g' | sed 's/([^)]*)//g' | xargs apt-get install -y --force-yes

deb: all
	touch tmp
	rm -rf tmp
	mkdir -p tmp/opt/scrappie/bin
	cp -pR DEBIAN tmp/
	$(SEDI) "s/SCRAPPIE_VERSION/$(SCRAPPIE_VERSION)/g"       tmp/DEBIAN/control
	chmod -R 0755 tmp/DEBIAN
	cp basecall *.py tmp/opt/scrappie/bin/
	dpkg -b tmp ont-scrappie-$(SCRAPPIE_VERSION).deb
