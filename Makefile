SHELL := /bin/bash
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TESTS_DIR   := $(PROJECT_DIR)/tests
BUILD_DIR   := $(TESTS_DIR)/results/build
OUTPUT      := $(PROJECT_DIR)/output.txt
VENV_ACT    := $(TESTS_DIR)/.venv/bin/activate
RUN_AUTOGRADER := \
    cd $(TESTS_DIR) && \
    source $(VENV_ACT) && \
    python autograder.py --student-dir $(PROJECT_DIR) --skip-build 2>&1

.PHONY: all build run rerun benchmark clean help
help:
	@echo "Available targets:"
	@echo "  build      - cmake --build the test harness with -j 2"
	@echo "  run        - build, then run the autograder once (creates output.txt)"
	@echo "  rerun      - build, then run the autograder once (appends to output.txt)"
	@echo "  benchmark  - build, then run the autograder 6 times (1 create + 5 append)"
	@echo "  clean      - remove output.txt and the test build directory"

all: benchmark
build:
	@if [ ! -f $(BUILD_DIR)/CMakeCache.txt ]; then \
	    cmake -S $(TESTS_DIR)/test_programs -B $(BUILD_DIR) \
	        -DSTUDENT_DIR=$(PROJECT_DIR) \
	        -DCMAKE_BUILD_TYPE=Release; \
	fi
	cmake --build $(BUILD_DIR) -j 2
run: build
	$(RUN_AUTOGRADER) | tee $(OUTPUT)
rerun: build
	$(RUN_AUTOGRADER) | tee -a $(OUTPUT)
benchmark: build
	$(RUN_AUTOGRADER) | tee    $(OUTPUT)
	$(RUN_AUTOGRADER) | tee -a $(OUTPUT)
	$(RUN_AUTOGRADER) | tee -a $(OUTPUT)
	$(RUN_AUTOGRADER) | tee -a $(OUTPUT)
	$(RUN_AUTOGRADER) | tee -a $(OUTPUT)
	$(RUN_AUTOGRADER) | tee -a $(OUTPUT)
clean:
	rm -f $(OUTPUT)
	rm -rf $(BUILD_DIR)