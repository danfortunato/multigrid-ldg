MKLROOT = /opt/intel/mkl

CC = gcc-7
CXX = g++-7
CFLAGS = -Wall -Wextra -Wno-int-in-bool-context -g -std=c++11 -O3 -m64
INCDIR = -I.. -I/opt/intel/mkl/include
LIBDIR = -L$(MKLROOT)/lib -Wl,-rpath,$(MKLROOT)/lib
LIB = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
BINDIR = bin
#OBJS = SparseBlockMatrix.o
#SRC = $(patsubst %.o,%.cpp,$(OBJS))
headers = SparseBlockMatrix.h
EXECS = $(BINDIR)/matrix_tests

all: $(MAKE) executables

executables: $(EXECS)

#depend: $(SRC)
#	$(CXX) $(INCDIR) -MM $(SRC) >Makefile.dep
#
#include Makefile.dep

$(BINDIR)/matrix_tests: matrix_tests.cpp
	$(CXX) $(CFLAGS) $(INCDIR) -o $@ $< $(LIBDIR) $(LIB)

%.o: %.cpp $(headers)
	$(CXX) $(CFLAGS) $(INCDIR) -c $< $(LIBDIR) $(LIB)

clean:
	rm -rf *.o *.dSYM/ $(EXECS)

#.PHONY: clean all executables depend
.PHONY: clean all executables
