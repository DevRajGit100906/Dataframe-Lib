# DataFrameLib

A columnar dataframe library written in C++17 on top of Apache Arrow,
supporting both eager and lazy evaluation modes, a fluent expression API,
and a rule-based query optimizer with eleven independent rewrite rules.

## Features

- **Eager mode** (`EagerDataFrame`) — every operation produces a
  materialized result immediately.
- **Lazy mode** (`LazyDataFrame`) — operations build a logical plan;
  execution is triggered only by `collect()`, `sink_csv()`, or
  `sink_parquet()`.
- **Expression system** with operator overloading: `col("x") + 5`,
  `col("dept") == "HR"`, `col("name").starts_with("A")`,
  `col("salary").mean()`, etc.
- **Joins** — inner, left, outer (right is also implemented for
  completeness but is not part of the required scope), with
  hash-based build on the smaller input.
- **Group-by aggregation** — sum, mean, count, min, max, available
  through multiple call shapes (string-pair vector, named-Expression
  vector, and a `map<string,string>` variant).
- **Query optimizer** — eleven rules: constant folding, expression
  simplification, filter elimination, conjunctive splitting,
  predicate pushdown, projection pushdown, limit pushdown, filter
  merging, projection merging, sort elimination, having-vs-where.
  Driver runs to fixpoint.
- **`explain()`** — renders the (optimized) logical plan to a PNG via
  Graphviz.
- **Strict null semantics** — null values are represented through
  Arrow null bitmaps; null in any operand yields null.
- **No `arrow::compute` or `arrow::acero`** — no compute kernels are
  called; storage and array construction use the core `arrow::Array`
  / `arrow::Builder` API only, and every per-row operation is a
  manual typed loop over Arrow buffers.

## Dependencies

- C++17 compiler (g++ 9+ or clang 10+)
- CMake ≥ 3.20
- Apache Arrow C++ (development headers)
- Apache Parquet C++ (development headers)
- Graphviz (`libcgraph`, `libgvc`)

### Installing dependencies on Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake \
    libarrow-dev libparquet-dev \
    libgraphviz-dev
```

## Building

From the project root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
```

The build produces `build/libdataframelib.a` (a static library) and a
set of CMake artifacts under `build/`. `-j 2` limits parallel compile
jobs; on memory-constrained machines (≤ 4 GB RAM, e.g. WSL with default
settings) this avoids out-of-memory thrashing during the simultaneous
compilation of many test programs.

## How to run

If you have placed your test programs under `tests/test_programs/`
(each one a `.cpp` file with a `main()` taking `<input_dir>` and
`<output_dir>` arguments) and the autograder harness under
`tests/`, the bundled `Makefile` drives the entire build-and-run
flow from the project root.

The autograder is a Python script (`tests/autograder.py`) and runs
inside a virtualenv at `tests/.venv/`. Create it once before the
first `make run` / `make benchmark`:

```bash
python3 -m venv tests/.venv
tests/.venv/bin/pip install -r tests/requirements.txt
```

This installs `pandas`, `pyarrow`, and `numpy` (used by the
autograder to verify each test's CSV output against a reference).
After that, the venv is reused across runs.

```bash
make build       # builds libdataframelib.a + every test program
make run         # builds, then runs all 17 test suites once
make benchmark   # builds, then runs the suite 6 times (R1 = cold; R2–R6 = timed)
make clean       # removes output.txt and the test build directory
```

`make run` writes a single `output.txt` at the project root.
`make benchmark` overwrites with the first run and appends the
remaining five, which is the procedure used to obtain the median
timings reported in the design document. The first run (R1) is a
cold run and is discarded; the medians are taken over R2–R6.

To run a single test program directly (e.g. while iterating on one
operation), invoke its binary from the test build directory:

```bash
./tests/results/build/test_lazy           tests/results/test_data  tests/results/test_outputs/test_lazy
./tests/results/build/test_performance    tests/results/test_data  tests/results/test_outputs/test_performance
```

Each test program takes two arguments: the input directory containing
the CSV/Parquet fixtures and an output directory for the result CSVs.

To run a tiny program of your own against the library, compile it
against the static library produced by `cmake --build build`:

```bash
g++ -std=c++17 -O2 -Iinclude my_app.cpp build/libdataframelib.a \
    -larrow -lparquet -lcgraph -lgvc -o my_app
./my_app
```

The `lazy.explain(path, optimized_plan)` call renders a side-by-side
PNG of the unoptimized and optimized plan via Graphviz; the
unoptimized side is always drawn, and the optimized side is drawn
when an `optimized_plan` produced by `QueryOptimizer::optimize` is
passed in.

## Using the library

Link against `dataframelib` and include the umbrella header:

```cpp
#include <dataframelib/dataframelib.h>
using namespace dataframelib;

int main() {
    // Eager mode
    auto df = read_csv("data.csv");
    auto result = df.filter(col("age") > 30)
                    .select({"name", "salary"});
    result.write_csv("output.csv");

    // Lazy mode
    auto lazy = scan_csv("data.csv")
                    .filter(col("age") > 30)
                    .group_by({"dept"})
                    .aggregate({{"avg_sal", col("salary").mean()}});
    lazy.explain("plan.png");      // render optimized plan
    auto out = lazy.collect();     // execute
    out.write_csv("agg.csv");
}
```

In your own `CMakeLists.txt`:

```cmake
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
add_subdirectory(path/to/dataframelib)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE dataframelib)
```

## Project layout

```text
project/
├── CMakeLists.txt                      Top-level build
├── include/dataframelib/
│   ├── dataframelib.h                  Umbrella header
│   ├── eager_dataframe.h               EagerDataFrame, GroupDataFrame
│   ├── lazy_dataframe.h                LazyDataFrame, LazyGroupDataFrame
│   ├── expr.h                          Expression AST + operator overloads
│   ├── kernel.h                        Manual kernels (templates)
│   ├── logical_plan.h                  Plan node hierarchy
│   └── query_optimizer.h               Rule interface + 11 rules
├── src/
│   ├── eager_dataframe.cpp             Eager evaluator
│   ├── lazy_dataframe.cpp              Lazy plan builder + collect()
│   ├── expr.cpp                        AST evaluation
│   ├── kernel.cpp                      Kernel function implementations
│   ├── logical_plan.cpp                Plan node execute() implementations
│   └── query_optimizer.cpp             Rule implementations + driver
└── report.tex                          Design document (build with pdflatex)
```

## API surface

### I/O

| Function | Description |
| --- | --- |
| `read_csv(path)` | Load CSV into an `EagerDataFrame` |
| `read_parquet(path)` | Load Parquet into an `EagerDataFrame` |
| `scan_csv(path)` | Create a `LazyDataFrame` from CSV |
| `scan_parquet(path)` | Create a `LazyDataFrame` from Parquet |
| `df.write_csv(path)` | Write an `EagerDataFrame` to CSV |
| `df.write_parquet(path)` | Write an `EagerDataFrame` to Parquet |
| `ldf.sink_csv(path)` | Execute lazy plan and write to CSV |
| `ldf.sink_parquet(path)` | Execute lazy plan and write to Parquet |
| `from_columns(columns)` | Build a DataFrame from columns; accepts either `vector<pair<string, shared_ptr<arrow::Array>>>` or `map<string, shared_ptr<arrow::Array>>` |

### DataFrame operations (both eager and lazy)

| Method | Description |
| --- | --- |
| `select(columns)` | Project a subset of named columns |
| `select_exprs(exprs)` | Project a list of expression columns |
| `filter(predicate)` | Filter rows by a Boolean predicate |
| `with_column(name, expr)` | Add or replace a column |
| `group_by(keys)` | Group by one or more columns |
| `aggregate(...)` | Aggregate grouped data (multiple call shapes) |
| `join(other, on, how)` | Inner / left / outer join (right also supported, not in required scope) |
| `sort(columns, asc)` | Sort rows |
| `head(n)` | First *n* rows |
| `collect()` | Execute the lazy plan |

### Expression API

| Construct | Description |
| --- | --- |
| `col(name)` | Column reference |
| `lit(value)` | Typed literal (int, double, string, bool) |
| `expr.alias(name)` | Rename result |
| `+ - * / %` | Element-wise arithmetic (with primitive overloads) |
| `== != < <= > >=` | Comparisons (with primitive overloads) |
| `&`, `\|`, `~` | Boolean AND, OR, NOT |
| `expr.is_null()` / `expr.is_not_null()` | Null checks |
| `expr.abs()` | Absolute value |
| `expr.length()` | String length |
| `expr.contains(s) / starts_with(s) / ends_with(s)` | Substring / prefix / suffix |
| `expr.to_lower() / to_upper()` | Case conversion |
| `expr.sum() / mean() / count() / min() / max()` | Aggregations |

## Design highlights

- **Shared kernel layer.** Lazy plan nodes call the same kernels as the
  eager API, so any kernel-level fix automatically improves both paths.
- **Hash join with smaller-side build.** The smaller input always becomes
  the build side; a `swapped` flag preserves correct left/right
  semantics for asymmetric joins.
- **`std::stable_sort` over index permutations.** No row copies happen
  inside the sort; only the final reorder-and-build step touches data.
- **Column-level type dispatch.** Each kernel inspects the input
  `arrow::Array` type once and then runs a typed inner loop.
- **Fixpoint optimizer.** Rules can enable each other — pushdown can
  reveal new constant-folding opportunities, etc. — so the driver
  re-runs the rule sequence until a full pass produces no changes.
- **Optimizer rules are pure functions.** Each rule maps `LogicalPlan`
  to `LogicalPlan` without mutating state, allowing free composition.
- **Graphviz used in-process** via `libcgraph` / `libgvc`; no shell-out.

## Build artefacts and submission hygiene

Before packaging, remove generated artefacts:

```bash
rm -rf build/
rm -rf tests/results/
rm -rf tests/.venv/
find . -type d -name __pycache__ -exec rm -rf {} +
```

Compiled binaries, CMake caches, generated test data, the Python
virtualenv, and any `__pycache__` directories left by the autograder
are not part of the deliverable. The `.gitignore` already excludes
them; run `git status` before tarring to verify a clean tree.
