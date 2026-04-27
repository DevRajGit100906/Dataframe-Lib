#include "dataframelib/lazy_dataframe.h"
#include "dataframelib/query_optimizer.h"
#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>
#include <sstream>
#include <fstream>
#include <stdexcept>

namespace dataframelib {
    LazyDataFrame::LazyDataFrame(std::shared_ptr<LogicalPlan> plan) : plan_(std::move(plan)) {};
    LazyGroupDataFrame::LazyGroupDataFrame(std::shared_ptr<LogicalPlan> plan, std::vector<std::string> keys)
                                                        : plan_(std::move(plan)), keys_(std::move(keys)) {};
    const std::shared_ptr<LogicalPlan>& LazyDataFrame::plan() const { 
        return plan_; 
    };
    EagerDataFrame LazyDataFrame::collect() const {
        auto optimized = QueryOptimizer::optimize(plan_);
        return optimized->execute();
    };
    LazyDataFrame LazyDataFrame::select(std::vector<std::string> columns) const {
        return LazyDataFrame(std::make_shared<ProjectPlan>(plan_, std::move(columns)));
    };
    LazyDataFrame LazyDataFrame::select_exprs(std::vector<Expression> exprs) const {
        return LazyDataFrame(std::make_shared<ProjectPlan>(plan_, exprs));
    };
    LazyDataFrame LazyDataFrame::filter(Expression predicate) const {
        auto new_node = std::make_shared<FilterPlan>(plan_, predicate);
        return LazyDataFrame(new_node);
    };
    LazyDataFrame LazyDataFrame::with_column(std::string name, Expression expr) const {
        auto new_node = std::make_shared<WithColumnPlan>(plan_, std::move(name), std::move(expr));
        return LazyDataFrame(new_node);
    };
    LazyDataFrame LazyDataFrame::agg(std::vector<Expression> aggr_exprs) const {
        auto new_node = std::make_shared<AggregatePlan>(plan_, std::vector<std::string>{}, std::move(aggr_exprs));
        return LazyDataFrame(new_node);
    };
    LazyDataFrame LazyDataFrame::join(const LazyDataFrame& other, const std::vector<std::string>& on, const std::string& how) const {
        auto new_node = std::make_shared<JoinPlan>(plan_, other.plan_, on, how);
        return LazyDataFrame(new_node);
    };
    LazyDataFrame LazyDataFrame::head(int64_t n) const {
        if (n < 0) {
            throw std::invalid_argument("Number of rows for head() must be non-negative");
        };
        auto new_node = std::make_shared<LimitPlan>(plan_, n);
        return LazyDataFrame(new_node);
    };
    LazyGroupDataFrame LazyDataFrame::group_by(std::vector<std::string> keys) const {
        return LazyGroupDataFrame(plan_, keys);
    };
    LazyDataFrame LazyDataFrame::sort(std::vector<std::string> columns, bool ascending) const {
        auto new_node = std::make_shared<SortPlan>(plan_, columns, ascending);
        return LazyDataFrame(new_node);
    };
    void LazyDataFrame::sink_csv(const std::string& path) const {
        collect().write_csv(path);
    };
    void LazyDataFrame::sink_parquet(const std::string& path) const {
        collect().write_parquet(path);
    };
    void LazyDataFrame::explain(const std::string& path, std::shared_ptr<LogicalPlan> optimized_plan) const {
        std::ostringstream dot_stream;
        dot_stream << "digraph G {\n";
        int node_counter = 0;
        dot_stream << "  subgraph cluster_unoptimized {\n";
        dot_stream << "    label=\"Unoptimized Plan\";\n";
        int root_id = 0;
        plan_->to_graphviz(dot_stream, node_counter, root_id);
        dot_stream << "  }\n";
        if (optimized_plan != nullptr) {
            dot_stream << "  subgraph cluster_optimized {\n";
            dot_stream << "    label=\"Optimized Plan\";\n";
            int opt_root_id = 0;
            optimized_plan->to_graphviz(dot_stream, node_counter, opt_root_id);
            dot_stream << "  }\n";
        };
        dot_stream << "}\n";
        std::string dot_str = dot_stream.str();
        GVC_t* gvc = gvContext();
        if (!gvc) {
            throw std::runtime_error("Failed to initialize Graphviz context.");
        };
        Agraph_t* g = agmemread(const_cast<char*>(dot_str.c_str()));
        if (!g) {
            gvFreeContext(gvc);
            throw std::runtime_error("Graphviz failed to parse the DOT string.");
        };
        int layout_status = gvLayout(gvc, g, "dot");
        if (layout_status != 0) {
            agclose(g);
            gvFreeContext(gvc);
            throw std::runtime_error("Graphviz failed to layout the graph.");
        };
        int render_status = gvRenderFilename(gvc, g, "png", path.c_str());
        gvFreeLayout(gvc, g);
        agclose(g);
        gvFreeContext(gvc);
        if (render_status != 0) {
            throw std::runtime_error("Graphviz failed to render the graph to file.");
        };
        std::ifstream check(path);
        if (!check.good()) {
            throw std::runtime_error("Graphviz failed to write the output file: " + path);
        };
    };
    LazyDataFrame LazyGroupDataFrame::agg(std::vector<Expression> aggr_exprs) const {
        auto new_node = std::make_shared<AggregatePlan>(plan_, keys_, std::move(aggr_exprs));
        return LazyDataFrame(new_node);
    };
    LazyDataFrame LazyDataFrame::aggregate(std::vector<std::pair<std::string, std::string>> agg_list) const {
        return group_by({}).aggregate(std::move(agg_list));
    };
    LazyDataFrame LazyDataFrame::aggregate_map(std::map<std::string, std::string> agg_map) const {
        return group_by({}).aggregate_map(std::move(agg_map));
    };
    LazyDataFrame LazyGroupDataFrame::aggregate_map(std::map<std::string, std::string> agg_map) const {
        std::vector<std::pair<std::string, std::string>> agg_list(agg_map.begin(), agg_map.end());
        return aggregate(std::move(agg_list));
    };
    LazyDataFrame LazyDataFrame::aggregate(std::vector<std::pair<std::string, Expression>> named_aggs) const {
        return group_by({}).aggregate(std::move(named_aggs));
    };
    LazyDataFrame LazyGroupDataFrame::aggregate(std::vector<std::pair<std::string, Expression>> named_aggs) const {
        std::vector<Expression> exprs;
        exprs.reserve(named_aggs.size());
        for (const auto& [name, expr] : named_aggs) {
            exprs.push_back(expr.alias(name));
        }
        return agg(std::move(exprs));
    };
    LazyDataFrame LazyGroupDataFrame::aggregate(std::vector<std::pair<std::string, std::string>> agg_list) const {
        std::vector<Expression> exprs;
        for (const auto& [col_name, fn_name] : agg_list) {
            Expression base = col(col_name);
            Expression agg_expr;
            if (fn_name == "sum") agg_expr = base.sum();
            else if (fn_name == "mean") agg_expr = base.mean();
            else if (fn_name == "count") agg_expr = base.count();
            else if (fn_name == "min") agg_expr = base.min();
            else if (fn_name == "max") agg_expr = base.max();
            else throw std::invalid_argument("Unknown aggregate function: " + fn_name);
            exprs.push_back(agg_expr.alias(col_name + "_" + fn_name));
        };
        return agg(std::move(exprs));
    };
    LazyDataFrame scan_csv(const std::string& path) {
        return LazyDataFrame(std::make_shared<ScanCsvPlan>(path));
    };
    LazyDataFrame scan_parquet(const std::string& path) {
        return LazyDataFrame(std::make_shared<ScanParquetPlan>(path));
    };
};