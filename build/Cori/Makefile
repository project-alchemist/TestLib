include $(ELEMENTAL_PATH)/conf/ElVars

SRC_PATH = $(TESTLIB_PATH)/src/main
TARGET_PATH = $(TESTLIB_PATH)/target

# Put Elemental's CXXFLAGS in front so ours can override them     "-I$(SRC_PATH)/ml" "-I$(SRC_PATH)/ml/clustering" "-I$(SRC_PATH)/nla"
CXXFLAGS += $(EL_COMPILE_FLAGS) -fdiagnostics-color=always
CXXFLAGS += -Wno-unused -Wno-reorder -std=c++11 -fopenmp
CXXFLAGS += "-I$(SRC_PATH)/include" "-I$(SRC_PATH)/utility"
CXXFLAGS += "-I$(ELEMENTAL_PATH)/include" "-I$(SPDLOG_PATH)/include" "-I$(EIGEN3_PATH)/include" "-I$(ARPACK_PATH)/include"
CXXFLAGS += -fPIC

LDFLAGS += "-L$(ELEMENTAL_PATH)/lib64" "-Wl,-rpath,$(ELEMENTAL_PATH)/lib64"
LDFLAGS += "-L$(EL_LIB)" "-Wl,-rpath,$(EL_LIB)" $(EL_LIBS)
LDFLAGS += "-Wl,-rpath,$(SPDLOG_PATH)/lib"
LDFLAGS += "-Wl,-rpath,$(ARPACK_PATH)/lib"
LDFLAGS += -Wl,-rpath,/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin/ -L/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin/ -lirc

#MODULES   := main main/ml/clustering main/nla
MODULES   := main
SRC_DIR   := $(addprefix $(TESTLIB_PATH)/src/,$(MODULES))
BUILD_DIR := $(addprefix $(TESTLIB_PATH)/target/,$(MODULES))

SRC       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
OBJ       := $(patsubst src/%.cpp,target/%.o,$(SRC))

vpath %.cpp $(SRC_DIR)

define make-goal
$1/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $$< -o $$@
endef

.PHONY: default

default: checkdirs $(TARGET_PATH)/testlib.so

$(TARGET_PATH)/testlib.so: $(OBJ)
	$(CXX) -shared $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))
