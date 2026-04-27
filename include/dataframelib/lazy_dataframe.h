#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "expr.h"
#include "logical_plan.h"

namespace dataframelib {
    class LazyGroupDataFrame;
    class LazyDataFrame {
        public:
            explicit LazyDataFrame(std::shared_ptr<LogicalPlan> plan);
            const std::shared_ptr<LogicalPlan>& plan() const;
            EagerDataFrame collect() const;
            LazyDataFrame select(std::vector<std::string> columns) const;
            LazyDataFrame select_exprs(std::vector<Expression> exprs) const;
            LazyDataFrame filter(Expression predicate) const;
            LazyDataFrame with_column(std::string name, Expression expr) const;
            LazyDataFrame head(int64_t n) const;
            LazyDataFrame sort(std::vector<std::string> columns, bool ascending = true) const;
            LazyGroupDataFrame group_by(std::vector<std::string> keys) const;
            LazyDataFrame agg(std::vector<Expression> aggr_exprs) const;
            LazyDataFrame aggregate(std::vector<std::pair<std::string, std::string>> agg_list) const;
            LazyDataFrame aggregate(std::vector<std::pair<std::string, Expression>> named_aggs) const;
            LazyDataFrame aggregate_map(std::map<std::string, std::string> agg_map) const;
            LazyDataFrame join(const LazyDataFrame& other,
                const std::vector<std::string>& on, 
                const std::string& how = "inner") const;
            void sink_csv(const std::string& path) const;
            void sink_parquet(const std::string& path) const;
            void explain(const std::string& path, std::shared_ptr<LogicalPlan> optimized_plan = nullptr) const;
        private:
            std::shared_ptr<LogicalPlan> plan_;
    };
    class LazyGroupDataFrame {
        public:
            LazyGroupDataFrame(std::shared_ptr<LogicalPlan> plan, std::vector<std::string> keys);
            LazyDataFrame agg(std::vector<Expression> aggr_exprs) const;
            LazyDataFrame aggregate(std::vector<std::pair<std::string, std::string>> agg_list) const;
            LazyDataFrame aggregate(std::vector<std::pair<std::string, Expression>> named_aggs) const;
            LazyDataFrame aggregate_map(std::map<std::string, std::string> agg_map) const;
        private:
            std::shared_ptr<LogicalPlan> plan_;
            std::vector<std::string> keys_;
    };
    LazyDataFrame scan_csv(const std::string& path);
    LazyDataFrame scan_parquet(const std::string& path);
}