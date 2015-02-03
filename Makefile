all: driver1 driver2 driver3 driver4 driver5 bench_minimization

# for debug:
#CC = gcc -lm -W -Wall -pedantic -Wmissing-prototypes \
#	-Wredundant-decls -Wnested-externs -Wdisabled-optimization \
#	-ansi -g -fexceptions -Wno-parentheses -Wshadow -Wcast-align \
#	-Winline -Wstrict-prototypes -Wno-unknown-pragmas -g -lpthread

# for blas:
# CC = gcc -lm -O3 -lpthread

# for speed:
CC = g++ -lm -O3 -std=c++0x

OBJ = cg_descent.o

# uncomment this line if using, for example, goto2 blas
LINK = -I/home/sm958/Work/pele/source -lcblas

$(OBJ): $(INCLUDE)

bench_minimization: $(OBJ) bench_minimization.cpp
	$(CC) -o bench_minimization bench_minimization.cpp $(OBJ) $(LINK) /home/sm958/Work/pele/source/lbfgs.cpp
	
driver1: $(OBJ) driver1.c
	$(CC) -o driver1 driver1.c $(OBJ) $(LINK)

driver2: $(OBJ) driver2.c
	$(CC) -o driver2 driver2.c $(OBJ) $(LINK)

driver3: $(OBJ) driver3.c
	$(CC) -o driver3 driver3.c $(OBJ) $(LINK)

driver4: $(OBJ) driver4.c
	$(CC) -o driver4 driver4.c $(OBJ) $(LINK)

driver5: $(OBJ) driver5.c
	$(CC) -o driver5 driver5.c $(OBJ) $(LINK)

clean:
	rm *.o

purge:
	rm *.o driver1 driver2 driver3 driver4 driver5 bench_minimization
