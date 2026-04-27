#include "dataframelib/logical_plan.h"
#include "dataframelib/kernel.h"
#include <algorithm>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <queue>
#include <sstream>

namespace dataframelib {
    ScanCsvPlan::ScanCsvPlan(std::string path)
                    : path_(std::move(path)) {};
    EagerDataFrame ScanCsvPlan::execute() const {
        return read_csv(path_);
    };
    PlanKind ScanCsvPlan::kind() const { 
        return PlanKind::ScanCsv; 
    };
    std::vector<std::shared_ptr<LogicalPlan>> ScanCsvPlan::children() const {
        return {};
    };
    std::string ScanCsvPlan::to_string() const {
        return "ScanCsv(" + path_ + ")";
    };
    void ScanCsvPlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
    };
    std::shared_ptr<arrow::Schema> ScanCsvPlan::output_schema() const {
        if (cached_schema_) return cached_schema_;
        cached_schema_ = read_csv(path_).schema();
        return cached_schema_;
    };
    ScanParquetPlan::ScanParquetPlan(std::string path)
                    : path_(std::move(path)) {};
    EagerDataFrame ScanParquetPlan::execute() const {
        return read_parquet(path_);
    };
    PlanKind ScanParquetPlan::kind() const { 
        return PlanKind::ScanParquet; 
    };
    std::vector<std::shared_ptr<LogicalPlan>> ScanParquetPlan::children() const {
        return {};
    };
    std::string ScanParquetPlan::to_string() const {
        return "ScanParquet(" + path_ + ")";
    };
    void ScanParquetPlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
    };
    std::shared_ptr<arrow::Schema> ScanParquetPlan::output_schema() const {
        if (cached_schema_) return cached_schema_;
        auto input_result = arrow::io::ReadableFile::Open(path_);
        if (!input_result.ok()) {
            throw std::runtime_error("Failed to open file: " + input_result.status().ToString());
        };
        auto input = *std::move(input_result);
        auto reader_result = parquet::arrow::OpenFile(input, arrow::default_memory_pool());
        if (!reader_result.ok()) {
            throw std::runtime_error("Failed to create Parquet reader: " + reader_result.status().ToString());
        };
        auto reader = std::move(reader_result).ValueOrDie();
        std::shared_ptr<arrow::Schema> schema;
        auto status = reader->GetSchema(&schema);
        if (!status.ok()) {
            throw std::runtime_error("Failed to read Parquet schema: " + status.ToString());
        };
        cached_schema_ = schema;
        return cached_schema_;
    };
    ProjectPlan::ProjectPlan(std::shared_ptr<LogicalPlan> child, std::vector<std::string> name_cols)
        : child_(std::move(child)), is_name_only_(true), name_cols_(std::move(name_cols)) {};
    ProjectPlan::ProjectPlan(std::shared_ptr<LogicalPlan> child, std::vector<Expression> expr_cols)
        : child_(std::move(child)), is_name_only_(false), expr_cols_(std::move(expr_cols)) {};
    EagerDataFrame ProjectPlan::execute() const {
        EagerDataFrame child_df = child_->execute();
        if (is_name_only_) {
            return child_df.select(name_cols_);
        } 
        else {
            return child_df.select_exprs(expr_cols_);
        };
    };
    PlanKind ProjectPlan::kind() const { 
        return PlanKind::Project; 
    };
    std::vector<std::shared_ptr<LogicalPlan>> ProjectPlan::children() const {
        return {child_};
    };
    std::string ProjectPlan::to_string() const {
        return is_name_only_ ? "Project(Names)" : "Project(Expressions)";
    };
    void ProjectPlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
        if (child_) {
            int child_id;
            child_->to_graphviz(out, node_counter, child_id);
            out << "node_" << my_id << " -> node_" << child_id << ";\n";
        };
    };
    std::shared_ptr<arrow::Schema> ProjectPlan::output_schema() const {
        auto child_schema = child_->output_schema();
        std::vector<std::shared_ptr<arrow::Field>> fields;
        if (is_name_only_) {
            fields.reserve(name_cols_.size());
            for (const auto& name : name_cols_) {
                auto field = child_schema->GetFieldByName(name);
                if (!field) {
                    throw std::invalid_argument("Column '" + name + "' does not exist in the table");
                };
                fields.push_back(field);
            };
        }
        else {
            fields.reserve(expr_cols_.size());
            for (size_t i = 0; i < expr_cols_.size(); ++i) {
                auto node_ptr = expr_cols_[i].node();
                auto type_result = node_ptr->infer_type(*child_schema);
                if (!type_result.ok()) {
                    throw std::runtime_error("Failed to infer type for expression '" + node_ptr->to_string() + "': " + type_result.status().ToString());
                };
                std::string name = "expr_" + std::to_string(i);
                if (node_ptr->kind() == ExprKind::Alias) {
                    name = std::static_pointer_cast<AliasExpr>(node_ptr)->output_name();
                }
                else if (node_ptr->kind() == ExprKind::Column) {
                    name = std::static_pointer_cast<ColumnExpr>(node_ptr)->name();
                };
                fields.push_back(arrow::field(name, type_result.ValueOrDie()));
            };
        };
        return arrow::schema(fields);
    };
    FilterPlan::FilterPlan(std::shared_ptr<LogicalPlan> child, Expression predicate)
                    : child_(std::move(child)), predicate_(std::move(predicate)) {};
    EagerDataFrame FilterPlan::execute() const {
        EagerDataFrame child_df = child_->execute();
        return child_df.filter(predicate_);
    };
    PlanKind FilterPlan::kind() const { 
        return PlanKind::Filter; 
    };
    std::vector<std::shared_ptr<LogicalPlan>> FilterPlan::children() const {
        return {child_};
    };
    std::string FilterPlan::to_string() const {
        return "Filter";
    };
    void FilterPlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
        if (child_) {
            int child_id;
            child_->to_graphviz(out, node_counter, child_id);
            out << "node_" << my_id << " -> node_" << child_id << ";\n";
        };
    };
    std::shared_ptr<arrow::Schema> FilterPlan::output_schema() const {
        return child_->output_schema();
    };
    WithColumnPlan::WithColumnPlan(std::shared_ptr<LogicalPlan> child, std::string name, Expression expr)
                            : child_(std::move(child)), name_(std::move(name)), expr_(std::move(expr)) {};
    EagerDataFrame WithColumnPlan::execute() const {
        EagerDataFrame child_df = child_->execute();
        return child_df.with_column(name_, expr_);
    };
    PlanKind WithColumnPlan::kind() const { 
        return PlanKind::WithColumn; 
    };
    std::vector<std::shared_ptr<LogicalPlan>> WithColumnPlan::children() const {
        return {child_};
    };
    std::string WithColumnPlan::to_string() const {
        return "WithColumn(name=" + name_ + ")";
    };
    void WithColumnPlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
        if (child_) {
            int child_id;
            child_->to_graphviz(out, node_counter, child_id);
            out << "node_" << my_id << " -> node_" << child_id << ";\n";
        };
    };
    std::shared_ptr<arrow::Schema> WithColumnPlan::output_schema() const {
        auto child_schema = child_->output_schema();
        auto type_result = expr_.node()->infer_type(*child_schema);
        if (!type_result.ok()) {
            throw std::runtime_error("Failed to infer type for new column '" + name_ + "': " + type_result.status().ToString());
        };
        auto new_field = arrow::field(name_, type_result.ValueOrDie());
        std::vector<std::shared_ptr<arrow::Field>> fields = child_schema->fields();
        int existing_idx = child_schema->GetFieldIndex(name_);
        if (existing_idx >= 0) {
            fields[existing_idx] = new_field;
        }
        else {
            fields.push_back(new_field);
        };
        return arrow::schema(fields);
    };
    AggregatePlan::AggregatePlan(std::shared_ptr<LogicalPlan> child,
        std::vector<std::string> group_keys_, 
        std::vector<Expression> aggr_exprs_)
        : child_(std::move(child)), group_keys_(std::move(group_keys_)), aggr_exprs_(std::move(aggr_exprs_)) {};
    EagerDataFrame AggregatePlan::execute() const {
        EagerDataFrame child_df = child_->execute();
        if (group_keys_.empty()) {
            return child_df.agg(aggr_exprs_);
        } 
        else {
            return child_df.group_by(group_keys_).agg(aggr_exprs_);
        };
    };
    PlanKind AggregatePlan::kind() const { 
        return PlanKind::Aggregate; 
    };
    std::vector<std::shared_ptr<LogicalPlan>> AggregatePlan::children() const {
        return {child_};
    };
    std::string AggregatePlan::to_string() const {
        return "Aggregate";
    };
    void AggregatePlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
        if (child_) {
            int child_id;
            child_->to_graphviz(out, node_counter, child_id);
            out << "node_" << my_id << " -> node_" << child_id << ";\n";
        };
    };
    std::shared_ptr<arrow::Schema> AggregatePlan::output_schema() const {
        auto child_schema = child_->output_schema();
        std::vector<std::shared_ptr<arrow::Field>> fields;
        fields.reserve(group_keys_.size() + aggr_exprs_.size());
        for (const auto& key : group_keys_) {
            auto field = child_schema->GetFieldByName(key);
            if (!field) {
                throw std::invalid_argument("Group key '" + key + "' does not exist in the table");
            };
            fields.push_back(field);
        };
        for (size_t i = 0; i < aggr_exprs_.size(); ++i) {
            ExprPtr node = aggr_exprs_[i].node();
            std::string output_name = "agg_" + std::to_string(i);
            if (node->kind() == ExprKind::Alias) {
                auto alias_node = std::static_pointer_cast<AliasExpr>(node);
                output_name = alias_node->output_name();
                node = alias_node->inner();
            };
            auto type_result = node->infer_type(*child_schema);
            if (!type_result.ok()) {
                throw std::runtime_error("Failed to infer type for aggregate expression: " + type_result.status().ToString());
            };
            fields.push_back(arrow::field(output_name, type_result.ValueOrDie()));
        };
        return arrow::schema(fields);
    };
    JoinPlan::JoinPlan(std::shared_ptr<LogicalPlan> left,
        std::shared_ptr<LogicalPlan> right, 
        std::vector<std::string> on, 
        std::string how)
        : left_(std::move(left)), right_(std::move(right)), on_(std::move(on)), how_(std::move(how)) {};
    EagerDataFrame JoinPlan::execute() const {
        EagerDataFrame left_df = left_->execute();
        EagerDataFrame right_df = right_->execute();
        return left_df.join(right_df, on_, how_);
    };
    PlanKind JoinPlan::kind() const { 
        return PlanKind::Join;
    };
    std::vector<std::shared_ptr<LogicalPlan>> JoinPlan::children() const {
        return {left_, right_};
    };
    std::string JoinPlan::to_string() const {
        return "Join(type=" + how_ + ")";
    };
    void JoinPlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
        if (left_) {
            int child_id;
            left_->to_graphviz(out, node_counter, child_id);
            out << "node_" << my_id << " -> node_" << child_id << ";\n";
        };
        if (right_) {
            int child_id;
            right_->to_graphviz(out, node_counter, child_id);
            out << "node_" << my_id << " -> node_" << child_id << ";\n";
        };
    };
    std::shared_ptr<arrow::Schema> JoinPlan::output_schema() const {
        auto left_schema = left_->output_schema();
        auto right_schema = right_->output_schema();
        std::vector<std::shared_ptr<arrow::Field>> fields;
        for (const auto& keyname : on_) {
            auto field = left_schema->GetFieldByName(keyname);
            if (!field) {
                throw std::invalid_argument("Join key '" + keyname + "' not found in left table");
            };
            fields.push_back(arrow::field(keyname, field->type()));
        };
        for (const auto& field : left_schema->fields()) {
            if (std::find(on_.begin(), on_.end(), field->name()) == on_.end()) {
                fields.push_back(arrow::field(field->name(), field->type()));
            };
        };
        for (const auto& field : right_schema->fields()) {
            if (std::find(on_.begin(), on_.end(), field->name()) != on_.end()) continue;
            std::string out_name = field->name();
            if (left_schema->GetFieldByName(field->name()) != nullptr) {
                out_name = field->name() + "_right";
            };
            fields.push_back(arrow::field(out_name, field->type()));
        };
        return arrow::schema(fields);
    };
    SortPlan::SortPlan(std::shared_ptr<LogicalPlan> child, std::vector<std::string> columns, bool ascending)
        : child_(std::move(child)), columns_(std::move(columns)), ascending_(ascending) {};
    EagerDataFrame SortPlan::execute() const {
        EagerDataFrame child_df = child_->execute();
        return child_df.sort(columns_, ascending_);
    };
    PlanKind SortPlan::kind() const { 
        return PlanKind::Sort; 
    };
    std::vector<std::shared_ptr<LogicalPlan>> SortPlan::children() const {
        return {child_};
    };
    std::string SortPlan::to_string() const {
        return "Sort";
    };
    void SortPlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
        if (child_) {
            int child_id;
            child_->to_graphviz(out, node_counter, child_id);
            out << "node_" << my_id << " -> node_" << child_id << ";\n";
        };
    };
    std::shared_ptr<arrow::Schema> SortPlan::output_schema() const {
        return child_->output_schema();
    };
    LimitPlan::LimitPlan(std::shared_ptr<LogicalPlan> child, int64_t n)
                                : child_(std::move(child)), n_(n) {};
    EagerDataFrame LimitPlan::execute() const {
        EagerDataFrame child_df = child_->execute();
        return child_df.head(n_);
    };
    PlanKind LimitPlan::kind() const { 
        return PlanKind::Limit; 
    };
    std::vector<std::shared_ptr<LogicalPlan>> LimitPlan::children() const {
        return {child_};
    };
    std::string LimitPlan::to_string() const {
        return "Limit(n=" + std::to_string(n_) + ")";
    };
    void LimitPlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
        if (child_) {
            int child_id;
            child_->to_graphviz(out, node_counter, child_id);
            out << "node_" << my_id << " -> node_" << child_id << ";\n";
        };
    };
    std::shared_ptr<arrow::Schema> LimitPlan::output_schema() const {
        return child_->output_schema();
    };
    EmptyTablePlan::EmptyTablePlan(std::shared_ptr<arrow::Schema> schema)
                                        : schema_(std::move(schema)) {};
    EagerDataFrame EmptyTablePlan::execute() const {
        std::vector<std::shared_ptr<arrow::ChunkedArray>> empty_arrays;
        for (const auto& field : schema_->fields()) {
            auto empty_array_result = arrow::MakeEmptyArray(field->type());
            empty_arrays.push_back(std::make_shared<arrow::ChunkedArray>(empty_array_result.ValueOrDie()));
        };
        auto empty_table = arrow::Table::Make(schema_, empty_arrays);
        return EagerDataFrame(empty_table);
    };
    PlanKind EmptyTablePlan::kind() const { 
        return PlanKind::EmptyTable; 
    };
    std::vector<std::shared_ptr<LogicalPlan>> EmptyTablePlan::children() const {
        return {};
    };
    std::string EmptyTablePlan::to_string() const {
        return "EmptyTable";
    };
    void EmptyTablePlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
    };
    std::shared_ptr<arrow::Schema> EmptyTablePlan::output_schema() const {
        return schema_;
    };
    TopKPlan::TopKPlan(std::shared_ptr<LogicalPlan> child, std::vector<std::string> columns, bool ascending, int64_t k)
        : child_(std::move(child)), columns_(std::move(columns)), ascending_(ascending), k_(k) {};

    EagerDataFrame TopKPlan::execute() const {
        EagerDataFrame df = child_->execute();
        if (df.num_rows() <= k_) {
            return df.sort(columns_, ascending_);
        };
        std::vector<std::shared_ptr<arrow::Array>> sort_arrays;
        sort_arrays.reserve(columns_.size());
        for (const auto& col_name : columns_) {
            if (!df.schema()->GetFieldByName(col_name))
                throw std::invalid_argument("Column '" + col_name + "' does not exist");
            auto chunked_col = df.table()->GetColumnByName(col_name);
            sort_arrays.push_back(arrow::Concatenate(chunked_col->chunks(), arrow::default_memory_pool()).ValueOrDie());
        };
        const bool ascending = ascending_;
        auto comp = [&sort_arrays, ascending](int64_t row_a, int64_t row_b) -> bool {
            for (const auto& col : sort_arrays) {
                if (col->IsNull(row_a) && col->IsNull(row_b)) continue;
                if (col->IsNull(row_a)) return ascending ? false : true;
                if (col->IsNull(row_b)) return ascending ? true : false;
                bool less, equal;
                switch (col->type_id()) {
                    case arrow::Type::INT64: {
                        auto va = std::static_pointer_cast<arrow::Int64Array>(col)->Value(row_a);
                        auto vb = std::static_pointer_cast<arrow::Int64Array>(col)->Value(row_b);
                        less = va < vb; equal = va == vb;
                        break;
                    }
                    case arrow::Type::DOUBLE: {
                        auto va = std::static_pointer_cast<arrow::DoubleArray>(col)->Value(row_a);
                        auto vb = std::static_pointer_cast<arrow::DoubleArray>(col)->Value(row_b);
                        less = va < vb; equal = va == vb;
                        break;
                    }
                    case arrow::Type::STRING: {
                        auto va = std::static_pointer_cast<arrow::StringArray>(col)->GetView(row_a);
                        auto vb = std::static_pointer_cast<arrow::StringArray>(col)->GetView(row_b);
                        less = va < vb; equal = va == vb;
                        break;
                    }
                    default: {
                        auto sa = col->GetScalar(row_a).ValueOrDie();
                        auto sb = col->GetScalar(row_b).ValueOrDie();
                        if (sa->Equals(*sb)) continue;
                        bool less_str = sa->ToString() < sb->ToString();
                        return ascending ? less_str : !less_str;
                    }
                };
                if (equal) continue;
                return ascending ? less : !less;
            };
            return false;
        };
        std::priority_queue<int64_t, std::vector<int64_t>, decltype(comp)> heap(comp);
        for (int64_t i = 0; i < df.num_rows(); ++i) {
            heap.push(i);
            if (static_cast<int64_t>(heap.size()) > k_) {
                heap.pop();
            };
        };
        std::vector<int64_t> top_k_indices(k_);
        for (int64_t i = k_ - 1; i >= 0; --i) {
            top_k_indices[i] = heap.top();
            heap.pop();
        };
        auto top_k_table = df.table();
        std::vector<std::shared_ptr<arrow::ChunkedArray>> top_k_arrays;
        for (const auto& chunked_col : top_k_table->columns()) {
            auto type = chunked_col->type();
            auto flat_col = arrow::Concatenate(chunked_col->chunks(), arrow::default_memory_pool()).ValueOrDie();
            std::shared_ptr<arrow::Array> new_col;
            std::unique_ptr<arrow::ArrayBuilder> builder;
            DFL_CHECK_OK(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder));
            for (const auto& idx : top_k_indices) {
                auto scalar_result = flat_col->GetScalar(idx);
                if (!scalar_result.ok()) throw std::runtime_error("Failed to get scalar value at row "
                                         + std::to_string(idx) + ": " + scalar_result.status().ToString());
                DFL_CHECK_OK(builder->AppendScalar(*scalar_result.ValueOrDie()));
            };
            DFL_CHECK_OK(builder->Finish(&new_col));
            top_k_arrays.push_back(std::make_shared<arrow::ChunkedArray>(new_col));
        };
        auto res = arrow::Table::Make(df.schema(), top_k_arrays);
        return EagerDataFrame(res);
    };
    PlanKind TopKPlan::kind() const {
        return PlanKind::TopK;
    };
    std::vector<std::shared_ptr<LogicalPlan>> TopKPlan::children() const {
        return {child_};
    };
    std::string TopKPlan::to_string() const {
        return "TopK(k=" + std::to_string(k_) + ")";
    };
    void TopKPlan::to_graphviz(std::ostringstream& out, int& node_counter, int& my_id) const {
        my_id = node_counter++;
        out << "node_" << my_id << " [label=\"" << this->to_string() << "\"];\n";
        if (child_) {
            int child_id;
            child_->to_graphviz(out, node_counter, child_id);
            out << "node_" << my_id << " -> node_" << child_id << ";\n";
        };
    };
    std::shared_ptr<arrow::Schema> TopKPlan::output_schema() const {
        return child_->output_schema();
    };
}