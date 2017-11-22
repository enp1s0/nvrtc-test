NVCC=nvcc
NVCCFLAGS= -arch=sm_60 -std=c++11 -lcuda -lnvrtc
CXX=g++
CXXFLAGS=-std=c++11
OBJDIR=obj
OBJLIST=cuda_common.o main.o
OBJS= $(addprefix $(OBJDIR)/, $(OBJLIST))
BIN=nvrtc-test

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

.SUFFIXES: .o .cu .cpp

.cu.cpp:
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@


clean:
	rm -rf $(OBJS)
	rm -rf $(BIN)
