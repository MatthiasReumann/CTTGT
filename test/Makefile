CCX = g++
CFLAGS = -std=c++17

INCLUDE += -I../include
INCLUDE += -I../src/external/marray/marray
INCLUDE += -I../src/external/blis/include/haswell
INCLUDE += -I../src/external/hptt/include

LIBS     = -lblis -lhptt

all: clean test

test:
	${CCX} ${CFLAGS} test.cpp ${INCLUDE} ${LIBS} -o $@.o

clean:
	rm test.o ||: