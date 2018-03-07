INTELROOT = /opt/intel
MKLROOT   = $(INTELROOT)/mkl

EXT         = lib
EIGENDIR    = /usr/local/include/eigen3
MGRIDGENDIR = $(EXT)/ParMGridGen-1.0

CC     = gcc-7
CXX    = g++-7
CFLAGS = -Wall -Wextra -Wno-int-in-bool-context -Wno-unused-parameter -std=c++17 -march=native
INCDIR = -I$(EIGENDIR) -I$(MKLROOT)/include -I$(MGRIDGENDIR)
LIBDIR = -L$(MGRIDGENDIR)
LIB    = -lmkl_intel_lp64 -lmkl_core -lm -ldl -lmgrid

ifndef BUILD
	BUILD = debug
endif

ifndef THREADS
	THREADS = 1
endif

# Type of build
ifeq ($(BUILD),debug)
	CFLAGS += -g -O0 -DDEBUG
else ifeq ($(BUILD),release)
	CFLAGS += -O3 -DNDEBUG
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

SRCDIR-cartesian   = $(addprefix $(SRCROOT)/cartesian/,$(MODULES))
BUILDDIR-cartesian = $(addprefix $(BUILDROOT)/cartesian/,$(MODULES))
INCDIR-cartesian   = $(INCDIR) $(addprefix -I,$(SRCDIR-cartesian))
BINDIR-cartesian   = $(BINDIR)/cartesian

SRCDIR-unstructured   = $(addprefix $(SRCROOT)/unstructured/,$(MODULES))
BUILDDIR-unstructured = $(addprefix $(BUILDROOT)/unstructured/,$(MODULES))
INCDIR-unstructured   = $(INCDIR) $(addprefix -I,$(SRCDIR-unstructured))
BINDIR-unstructured   = $(BINDIR)/unstructured

SRC-cartesian  = $(foreach sdir,$(SRCDIR-cartesian), $(wildcard $(sdir)/*.cpp))
OBJ-cartesian  = $(patsubst $(SRCROOT)/%.cpp,$(BUILDROOT)/%.o,$(SRC-cartesian))
SRC-unstructured  = $(foreach sdir,$(SRCDIR-unstructured), $(wildcard $(sdir)/*.cpp))
OBJ-unstructured  = $(patsubst $(SRCROOT)/%.cpp,$(BUILDROOT)/%.o,$(SRC-unstructured))

headers = common.h sparseblockmatrix.h element.h master.h mesh.h quadtree.h function.h ldgpoisson.h
EXECS  = cartesian/matrix_tests cartesian/dg_tests unstructured/dg_tests unstructured/l2_projection_test
EXECS := $(addprefix $(BINDIR)/,$(EXECS))

define cc-command-cartesian
	$(CXX) $(CFLAGS) $(INCDIR-cartesian) -o $@ $< $(LIBDIR) $(LIB)
endef

define cc-command-unstructured
	$(CXX) $(CFLAGS) $(INCDIR-unstructured) -o $@ $< $(LIBDIR) $(LIB)
endef

define make-goal-cartesian
$1/%.o: %.cpp
    $(CXX) $(CFLAGS) $(INCDIR-cartesian) -c $$< -o $$@ $(LIBDIR) $(LIB)
endef

define make-goal-unstructured
$1/%.o: %.cpp
    $(CXX) $(CFLAGS) $(INCDIR-unstructured) -c $$< -o $$@ $(LIBDIR) $(LIB)
endef

all: checkdirs $(MAKE) executables

executables: $(EXECS)

#depend: $(SRC)
#	$(CXX) $(IFLAGS) -MM $(SRC) >Makefile.dep
#
#include Makefile.dep

$(BINDIR)/cartesian/matrix_tests: $(TESTDIR)/cartesian/matrix_tests.cpp $(OBJ-cartesian)
	$(cc-command-cartesian)

$(BINDIR)/cartesian/dg_tests: $(TESTDIR)/cartesian/dg_tests.cpp $(OBJ-cartesian)
	$(cc-command-cartesian)

$(BINDIR)/unstructured/dg_tests: $(TESTDIR)/unstructured/dg_tests.cpp $(OBJ-unstructured)
	$(cc-command-unstructured)

$(BINDIR)/unstructured/basis_tests: $(TESTDIR)/unstructured/basis_tests.cpp $(OBJ-unstructured)
	$(cc-command-unstructured)

$(BINDIR)/unstructured/l2_projection_test: $(TESTDIR)/unstructured/l2_projection_test.cpp $(OBJ-unstructured)
	$(cc-command-unstructured)

checkdirs: $(BINDIR-cartesian) $(BINDIR-unstructured) $(BUILDDIR-cartesian) $(BUILDDIR-unstructured)

$(BINDIR-cartesian):
	@mkdir -p $@

$(BINDIR-unstructured):
	@mkdir -p $@

$(BUILDDIR-cartesian):
	@mkdir -p $@

$(BUILDDIR-unstructured):
	@mkdir -p $@

clean:
	rm -rf $(BUILDROOT)
	rm -rf $(BINDIR)

.PHONY: clean all checkdirs executables depend

	$(foreach bdir,$(BUILDDIR-cartesian),$(eval $(call make-goal-cartesian,$(bdir))))
	$(foreach bdir,$(BUILDDIR-unstructured),$(eval $(call make-goal-unstructured,$(bdir))))
