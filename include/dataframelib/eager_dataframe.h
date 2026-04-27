#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "expr.h"
#include <unordered_map>
#include <map>

namespace arrow {
    class Array;
    class Schema;
    class Table;
};
namespace dataframelib {
    enum class Side {
        Left, Right,
        Key
    };
    struct FieldSource {
        Side side;
        std::string src_name;
    };
    struct SortKey {
        arrow::Type::type type_id;
        const arrow::Array* array;
        const int64_t* index_64;
        const double* index_double;
        const arrow::StringArray* index_string;
    };
    class GroupDataFrame;
    class EagerDataFrame {
        public:
            EagerDataFrame();
            explicit EagerDataFrame(std::shared_ptr<arrow::Table> table);
            EagerDataFrame(const EagerDataFrame&) = default;
            EagerDataFrame(EagerDataFrame&&) noexcept = default;
            EagerDataFrame& operator=(const EagerDataFrame&) = default;
            EagerDataFrame& operator=(EagerDataFrame&&) noexcept = default;
            int64_t num_rows() const noexcept;
            int num_cols() const noexcept;
            int num_columns() const noexcept { return num_cols(); }
            std::shared_ptr<arrow::Schema> schema() const;
            std::vector<std::string> column_names() const;
            const std::shared_ptr<arrow::Table>& table() const noexcept;
            void print(int64_t max_rows = 10) const;
            static EagerDataFrame from_columns(
                const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& columns);
            static EagerDataFrame from_columns(
                const std::map<std::string, std::shared_ptr<arrow::Array>>& columns);
            void write_csv(const std::string& path) const;
            void write_parquet(const std::string& path, int64_t chunk_size = 64 * 1024) const;
            EagerDataFrame select(const std::vector<std::string>& columns) const;
            EagerDataFrame select_exprs(const std::vector<Expression>& exprs) const;
            EagerDataFrame filter(const Expression& predicate) const;
            EagerDataFrame with_column(const std::string& name, const Expression& expr) const;
            EagerDataFrame head(int64_t n = 10) const;
            EagerDataFrame sort(const std::vector<std::string>& columns, bool ascending = true) const;
            GroupDataFrame group_by(const std::vector<std::string>& keys) const;
            EagerDataFrame agg(const std::vector<Expression>& aggr_exprs) const;
            EagerDataFrame aggregate(const std::vector<std::pair<std::string, std::string>>& agg_list) const;
            EagerDataFrame aggregate(const std::vector<std::pair<std::string, Expression>>& named_aggs) const;
            EagerDataFrame aggregate_map(const std::map<std::string, std::string>& agg_map) const;
            EagerDataFrame join(const EagerDataFrame& other, const std::vector<std::string>& on, const std::string& how = "inner") const;
        private:
            std::shared_ptr<arrow::Table> table_;
    };
    class GroupDataFrame {
        public:
            GroupDataFrame(const EagerDataFrame& df, const std::vector<std::string>& keys);
            EagerDataFrame agg(const std::vector<Expression>& aggr_exprs) const;
            EagerDataFrame aggregate(const std::vector<std::pair<std::string, std::string>>& agg_list) const;
            EagerDataFrame aggregate(const std::vector<std::pair<std::string, Expression>>& named_aggs) const;
            EagerDataFrame aggregate_map(const std::map<std::string, std::string>& agg_map) const;
        private:
            std::shared_ptr<arrow::Table> table_;
            std::vector<std::string> group_keys;
    };
    std::string grouping_helper(int64_t row_index, const std::vector<std::shared_ptr<arrow::Array>>& flat_keys);
    EagerDataFrame read_csv(const std::string& path);
    EagerDataFrame read_parquet(const std::string& path);
    template <typename = void>
    inline EagerDataFrame from_columns(
        const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& columns) {
        return EagerDataFrame::from_columns(columns);
    };
    inline EagerDataFrame from_columns(
        const std::map<std::string, std::shared_ptr<arrow::Array>>& columns) {
        return EagerDataFrame::from_columns(columns);
    };
}