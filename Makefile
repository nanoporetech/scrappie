CC = gcc
INC=-I/usr/include/hdf5/serial
LIBS=/usr/lib/openblas-base/libblas.a -lhdf5_serial -lm 
CFLAGS = -fopenmp -Ofast  -march=native -ffast-math -std=gnu99 -DFAST_EXP
OBJDIR = obj

OBJECTS = read_events.o features.o util.o layers.o 

all: basecall_gru

%.o: %.c
	$(CC) $(INC) -c -o $@ $< $(CFLAGS)

basecall_gru: basecall_gru.o $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

basecall_lstm: basecall_lstm.o $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -f *.o basecall_gru basecall_lstm
