INTELROOT = /opt/intel
MKLROOT   = $(INTELROOT)/mkl

CC     = gcc-7
CXX    = g++-7
CFLAGS = -Wall -Wextra -Wno-int-in-bool-context -Wno-unused-parameter -g -std=c++17 -O3 -fopenmp
CFLAGS += -DNDEBUG
INCDIR = -I.. -I$(MKLROOT)/include
LIBDIR = -L$(INTELROOT)/lib -L$(MKLROOT)/lib -Wl,-rpath,$(MKLROOT)/lib,-rpath,$(INTELROOT)/lib
LIB    = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

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
