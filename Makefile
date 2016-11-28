MAJOR  ?= 0
MINOR  ?= 0
SUB    ?= 1
PATCH  ?= 1


CC     ?= gcc
LIBS    = -lblas -lhdf5 -lm  
CFLAGS  = -Wall -Wno-unused-function -Ofast  -fopenmp   -march=native -ffast-math -std=c99 -DUSE_SSE2
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
	$(SEDI) "s/MAJOR/$(MAJOR)/g"       tmp/DEBIAN/control
	$(SEDI) "s/MINOR/$(MINOR)/g"       tmp/DEBIAN/control
	$(SEDI) "s/PATCH/$(PATCH)/g"       tmp/DEBIAN/control
	$(SEDI) "s/SUB/$(SUB)/g"           tmp/DEBIAN/control
	chmod -R 0755 tmp/DEBIAN
	cp basecall *.py tmp/opt/scrappie/bin/
	dpkg -b tmp ont-scrappie-$(MAJOR).$(MINOR).$(PATCH)-$(SUB).deb
