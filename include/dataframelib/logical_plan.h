#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "eager_dataframe.h"
#include "expr.h"

namespace dataframelib {
    enum class PlanKind {
        ScanCsv, ScanParquet,
        Filter, Project,
        WithColumn, Aggregate,
        Join, Sort, Limit,
        EmptyTable, TopK
    };
    class LogicalPlan {
        public:
            LogicalPlan() = default;
            virtual ~LogicalPlan() = default;
            virtual EagerDataFrame execute() const = 0;
            virtual PlanKind kind() const = 0;
            virtual std::vector<std::shared_ptr<LogicalPlan>> children() const = 0;
            virtual std::string to_string() const = 0;
            virtual void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const = 0;
            virtual std::shared_ptr<arrow::Schema> output_schema() const = 0;
    };
    class ScanCsvPlan : public LogicalPlan {
        public:
            explicit ScanCsvPlan(std::string path);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
            const std::string& path() const { return path_; }
        private:
            std::string path_;
            mutable std::shared_ptr<arrow::Schema> cached_schema_;
    };
    class ScanParquetPlan : public LogicalPlan {
        public:
            explicit ScanParquetPlan(std::string path);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
            const std::string& path() const { return path_; }
        private:
            std::string path_;
            mutable std::shared_ptr<arrow::Schema> cached_schema_;
    };
    class ProjectPlan : public LogicalPlan {
        public:
            ProjectPlan(std::shared_ptr<LogicalPlan> child, std::vector<std::string> name_cols);
            ProjectPlan(std::shared_ptr<LogicalPlan> child, std::vector<Expression> expr_cols);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
            bool is_name_only() const { return is_name_only_; }
            const std::vector<std::string>& name_cols() const { return name_cols_; }
            const std::vector<Expression>& expr_cols() const { return expr_cols_; }
        private:
            std::shared_ptr<LogicalPlan> child_;
            std::vector<Expression> columns_;
            bool is_name_only_;
            std::vector<std::string> name_cols_;
            std::vector<Expression> expr_cols_;
    };
    class FilterPlan : public LogicalPlan {
        public:
            FilterPlan(std::shared_ptr<LogicalPlan> child, Expression predicate);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
            const Expression& predicate() const { return predicate_; }
        private:
            std::shared_ptr<LogicalPlan> child_;
            Expression predicate_;
    };
    class WithColumnPlan : public LogicalPlan {
        public:
            WithColumnPlan(std::shared_ptr<LogicalPlan> child, std::string name, Expression expr);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
            const std::string& name() const { return name_; }
            const Expression& expr() const { return expr_; }
        private:
            std::shared_ptr<LogicalPlan> child_;
            std::string name_;
            Expression expr_;
    };
    class AggregatePlan : public LogicalPlan {
        public:
            AggregatePlan(std::shared_ptr<LogicalPlan> child, std::vector<std::string> group_keys_, std::vector<Expression> aggr_exprs_);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
            const std::vector<std::string>& group_keys() const { return group_keys_; }
            const std::vector<Expression>& aggr_exprs() const { return aggr_exprs_; }
        private:
            std::shared_ptr<LogicalPlan> child_;
            std::vector<std::string> group_keys_;
            std::vector<Expression> aggr_exprs_;
    };
    class JoinPlan : public LogicalPlan {
        public:
            JoinPlan(std::shared_ptr<LogicalPlan> left, std::shared_ptr<LogicalPlan> right, std::vector<std::string> on, std::string how);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
            const std::vector<std::string>& on() const { return on_; }
            const std::string& how() const { return how_; }
        private:
            std::shared_ptr<LogicalPlan> left_;
            std::shared_ptr<LogicalPlan> right_;
            std::vector<std::string> on_;
            std::string how_;
    };
    class SortPlan : public LogicalPlan {
        public:
            SortPlan(std::shared_ptr<LogicalPlan> child, std::vector<std::string> columns, bool ascending);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
            const std::vector<std::string>& columns() const { return columns_; }
            bool ascending() const { return ascending_; }
        private:
            std::shared_ptr<LogicalPlan> child_;
            std::vector<std::string> columns_;
            bool ascending_;
    };
    class LimitPlan : public LogicalPlan {
        public:
            LimitPlan(std::shared_ptr<LogicalPlan> child, int64_t n);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
            int64_t n() const { return n_; }
        private:
            std::shared_ptr<LogicalPlan> child_;
            int64_t n_;
    };
    class EmptyTablePlan : public LogicalPlan {
        public:
            explicit EmptyTablePlan(std::shared_ptr<arrow::Schema> schema);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
        private:
            std::shared_ptr<arrow::Schema> schema_;
    };
    class TopKPlan : public LogicalPlan {
        public:
            TopKPlan(std::shared_ptr<LogicalPlan> child, std::vector<std::string> columns, bool ascending, int64_t k);
            EagerDataFrame execute() const override;
            PlanKind kind() const override;
            std::vector<std::shared_ptr<LogicalPlan>> children() const override;
            std::string to_string() const override;
            void to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const override;
            std::shared_ptr<arrow::Schema> output_schema() const override;
            const std::vector<std::string>& columns() const { return columns_; }
            bool ascending() const { return ascending_; }
            int64_t k() const { return k_; }
        private:
            std::shared_ptr<LogicalPlan> child_;
            std::vector<std::string> columns_;
            bool ascending_;
            int64_t k_;
    };
}