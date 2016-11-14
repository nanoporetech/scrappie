CC = gcc
INC=-I/usr/include/hdf5/serial
LIBS=/usr/lib/openblas-base/libblas.a -lhdf5_serial -lm 
CFLAGS = -fopenmp -Ofast  -march=native -ffast-math -std=gnu99 -DFAST_EXP
OBJDIR = obj

OBJECTS = basecall.o read_events.o features.o util.o layers.o 

%.o: %.c
	$(CC) $(INC) -c -o $@ $< $(CFLAGS)

basecall: $(OBJECTS)
	$(CC) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

clean:
	rm *.o basecall
