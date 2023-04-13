
CC=cc
CFLAGS=-O3 -fopenmp
LIBS=

mt-dgemm: mt-dgemm.c
	$(CC) $(CFLAGS) -o mt-dgemm mt-dgemm.c $(LIBS)

clean:
	rm -f mt-dgemm

