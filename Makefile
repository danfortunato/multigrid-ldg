INTELROOT = /opt/intel
MKLROOT   = $(INTELROOT)/mkl

CC     = gcc-7
CXX    = g++-7
CFLAGS = -Wall -Wextra -Wno-int-in-bool-context -Wno-unused-parameter -std=c++17 -O3 -march=native
INCDIR = -I.. -I$(MKLROOT)/include
LIBDIR =
LIB    = -lmkl_intel_lp64 -lmkl_core -lm -ldl

ifndef BUILD
	BUILD = debug
endif

ifndef THREADS
	THREADS = 1
endif

# Type of build
ifeq ($(BUILD),debug)
	CFLAGS += -g -DDEBUG
else ifeq ($(BUILD),release)
	CFLAGS += -DNDEBUG
endif

# Number of threads to use
ifeq ($(THREADS),1)
	CFLAGS += -DMKL_THREADING_LAYER=sequential -DMKL_NUM_THREADS=1 -DMKL_DYNAMIC="FALSE" -DMKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=1" -DOMP_NUM_THREADS=1 -DOMP_DYNAMIC="FALSE"
	LIB += -lmkl_sequential
else
	CFLAGS += -fopenmp -DMKL_NUM_THREADS=$(THREADS) -DMKL_DYNAMIC="TRUE" -DMKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=$(THREADS), MKL_DOMAIN_BLAS=$(THREADS)" -DOMP_NUM_THREADS=$(THREADS) -DOMP_DYNAMIC="TRUE"
	LIB += -lmkl_intel_thread -liomp5 -lpthread
endif

# MKL directories on Mac and Linux are different
OS := $(shell uname)
ifeq ($(OS),Darwin)
	LIBDIR += -L$(INTELROOT)/lib -L$(MKLROOT)/lib -Wl,-rpath,$(MKLROOT)/lib,-rpath,$(INTELROOT)/lib
else ifeq ($(OS),Linux)
	LIBDIR += -L$(INTELROOT)/lib/intel64 -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed
endif

BINDIR    = bin
BUILDROOT = build
TESTDIR   = tests
SRCROOT   = src
MODULES   = dg multigrid mesh util

SRCDIR    = $(addprefix $(SRCROOT)/,$(MODULES))
BUILDDIR  = $(addprefix $(BUILDROOT)/,$(MODULES))
INCDIR   += $(addprefix -I,$(SRCDIR))

VPATH = %.cpp $(SRCDIR)

SRC  = $(foreach sdir,$(SRCDIR), $(wildcard $(sdir)/*.cpp))
OBJ  = $(patsubst $(SRCROOT)/%.cpp,build/%.o,$(SRC))

headers = common.h sparseblockmatrix.h element.h master.h mesh.h quadtree.h function.h ldgpoisson.h
EXECS  = matrix_tests dg_tests
EXECS := $(addprefix $(BINDIR)/,$(EXECS))

define cc-command
	$(CXX) $(CFLAGS) $(INCDIR) -o $@ $< $(LIBDIR) $(LIB)
endef

define make-goal
$1/%.o: %.cpp
    $(CXX) $(CFLAGS) $(INCDIR) -c $$< -o $$@ $(LIBDIR) $(LIB)
endef

all: checkdirs $(MAKE) executables

executables: $(EXECS)

#depend: $(SRC)
#	$(CXX) $(IFLAGS) -MM $(SRC) >Makefile.dep
#
#include Makefile.dep

$(BINDIR)/matrix_tests: $(TESTDIR)/matrix_tests.cpp $(OBJ)
	$(cc-command)

$(BINDIR)/dg_tests: $(TESTDIR)/dg_tests.cpp $(OBJ)
	$(cc-command)

checkdirs: $(BINDIR) $(BUILDDIR)

$(BINDIR):
	@mkdir -p $@

$(BUILDDIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILDROOT)
	rm -rf $(BINDIR)

#.PHONY: clean all executables depend
.PHONY: clean all checkdirs executables depend

	$(foreach bdir,$(BUILDDIR),$(eval $(call make-goal,$(bdir))))
