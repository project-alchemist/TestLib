include $(ELEMENTAL_PATH)/conf/ElVars

SRC_PATH = $(TESTLIB_PATH)/src/main
TARGET_PATH = $(TESTLIB_PATH)/target

CXXFLAGS += $(EL_COMPILE_FLAGS) -fdiagnostics-color=always
CXXFLAGS += -Wno-unused -Wno-reorder -std=c++11 -fext-numeric-literals -fopenmp
CXXFLAGS += "-I$(SRC_PATH)/include" "-I$(SRC_PATH)/utility"
CXXFLAGS += "-I$(ELEMENTAL_PATH)/include" "-I$(SPDLOG_PATH)/include" "-I$(EIGEN3_PATH)/include" "-I$(ARPACK_PATH)/include"
CXXFLAGS += -fPIC

LDFLAGS += "-L$(EL_LIB)" "-Wl,-rpath,$(EL_LIB)" $(EL_LIBS)
LDFLAGS += "-Wl,-rpath,$(SPDLOG_PATH)/lib"
LDFLAGS += "-Wl,-rpath,$(ARPACK_PATH)/lib"
LDFLAGS +=  -lmpi

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
