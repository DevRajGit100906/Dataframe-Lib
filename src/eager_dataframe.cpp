#include "dataframelib/eager_dataframe.h"
#include "dataframelib/kernel.h"
#include <algorithm>
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/pretty_print.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_set>
#include <stdexcept>

namespace dataframelib {
    inline void filter_column_typed (
        const std::shared_ptr<arrow::ChunkedArray>& column,
        const arrow::BooleanArray& mask,
        arrow::ArrayBuilder* builder) {
        int64_t index = 0;
        switch (column->type()->id()) {
            case arrow::Type::INT64: {
                auto int_builder = static_cast<arrow::Int64Builder*>(builder);
                for (const auto& chunk : column->chunks()) {
                    auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
                    const int64_t* vals = arr->raw_values();
                    const int64_t length = chunk->length();
                    for (int64_t r = 0; r < length; ++r, ++index) {
                        if (mask.IsNull(index) || !mask.Value(index)) continue;
                        if (arr->IsNull(r)) int_builder->UnsafeAppendNull();
                        else int_builder->UnsafeAppend(vals[r]);
                    };
                };
                break;
            };
            case arrow::Type::DOUBLE: {
                auto* db = static_cast<arrow::DoubleBuilder*>(builder);
                for (const auto& chunk : column->chunks()) {
                    auto arr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                    const double* vals = arr->raw_values();
                    const int64_t n = chunk->length();
                    for (int64_t r = 0; r < n; ++r, ++index) {
                        if (mask.IsNull(index) || !mask.Value(index)) continue;
                        if (arr->IsNull(r)) db->UnsafeAppendNull();
                        else db->UnsafeAppend(vals[r]);
                    };
                };
                break;
            };
            case arrow::Type::BOOL: {
                auto* bb = static_cast<arrow::BooleanBuilder*>(builder);
                for (const auto& chunk : column->chunks()) {
                    auto arr = std::static_pointer_cast<arrow::BooleanArray>(chunk);
                    const int64_t n = chunk->length();
                    for (int64_t r = 0; r < n; ++r, ++index) {
                        if (mask.IsNull(index) || !mask.Value(index)) continue;
                        if (arr->IsNull(r)) bb->UnsafeAppendNull();
                        else bb->UnsafeAppend(arr->Value(r));
                    };
                };
                break;
            };
            case arrow::Type::STRING: {
                auto* sb = static_cast<arrow::StringBuilder*>(builder);
                for (const auto& chunk : column->chunks()) {
                    auto arr = std::static_pointer_cast<arrow::StringArray>(chunk);
                    const int64_t n = chunk->length();
                    for (int64_t r = 0; r < n; ++r, ++index) {
                        if (mask.IsNull(index) || !mask.Value(index)) continue;
                        if (arr->IsNull(r))
                            DFL_CHECK_OK(sb->AppendNull());
                        else
                            DFL_CHECK_OK(sb->Append(arr->GetView(r)));
                    };
                };
                break;
            };
            default: {
                for (const auto& chunk : column->chunks()) {
                    const int64_t n = chunk->length();
                    for (int64_t r = 0; r < n; ++r, ++index) {
                        if (mask.IsNull(index) || !mask.Value(index)) continue;
                        auto sc = chunk->GetScalar(r).ValueOrDie();
                        DFL_CHECK_OK(builder->AppendScalar(*sc));
                    };
                };
                break;
            };
        };
    };
    inline void gather_column_typed(
        const std::shared_ptr<arrow::Array>& flat_col,
        const std::vector<int64_t>& indices,
        arrow::ArrayBuilder* builder) {
        switch (flat_col->type()->id()) {
            case arrow::Type::INT64: {
                auto* ib = static_cast<arrow::Int64Builder*>(builder);
                auto arr = std::static_pointer_cast<arrow::Int64Array>(flat_col);
                const int64_t* vals = arr->raw_values();
                for (int64_t idx : indices) {
                    if (idx < 0 || arr->IsNull(idx)) ib->UnsafeAppendNull();
                    else ib->UnsafeAppend(vals[idx]);
                };
                break;
            };
            case arrow::Type::DOUBLE: {
                auto* db = static_cast<arrow::DoubleBuilder*>(builder);
                auto arr = std::static_pointer_cast<arrow::DoubleArray>(flat_col);
                const double* vals = arr->raw_values();
                for (int64_t idx : indices) {
                    if (idx < 0 || arr->IsNull(idx)) db->UnsafeAppendNull();
                    else db->UnsafeAppend(vals[idx]);
                };
                break;
            };
            case arrow::Type::BOOL: {
                auto* bb = static_cast<arrow::BooleanBuilder*>(builder);
                auto arr = std::static_pointer_cast<arrow::BooleanArray>(flat_col);
                for (int64_t idx : indices) {
                    if (idx < 0 || arr->IsNull(idx)) bb->UnsafeAppendNull();
                    else bb->UnsafeAppend(arr->Value(idx));
                };
                break;
            };
            case arrow::Type::STRING: {
                auto* sb = static_cast<arrow::StringBuilder*>(builder);
                auto arr = std::static_pointer_cast<arrow::StringArray>(flat_col);
                for (int64_t idx : indices) {
                    if (idx < 0 || arr->IsNull(idx))
                        DFL_CHECK_OK(sb->AppendNull());
                    else
                        DFL_CHECK_OK(sb->Append(arr->GetView(idx)));
                };
                break;
            };
            default: {
                for (int64_t idx : indices) {
                    if (idx < 0) {
                        DFL_CHECK_OK(builder->AppendNull());
                    } 
                    else {
                        auto sc = flat_col->GetScalar(idx).ValueOrDie();
                        DFL_CHECK_OK(builder->AppendScalar(*sc));
                    };
                };
                break;
            };
        };
    };
    std::string grouping_helper(int64_t row_index, const std::vector<std::shared_ptr<arrow::Array>>& flat_keys) {
        std::string result;
        for (size_t i = 0; i < flat_keys.size(); i++) {
            auto scalar = flat_keys[i]->GetScalar(row_index).ValueOrDie();
            if (scalar->is_valid) {
                if (scalar->type->id() == arrow::Type::BOOL) {
                    result += "1:" + std::string(std::static_pointer_cast<arrow::BooleanScalar>(scalar)->value ? "1" : "0");
                } 
                else {
                    std::string s = scalar->ToString();
                    result += std::to_string(s.size()) + ":" + s;
                };
            } 
            else {
                result += "-1:?"; 
            };
            if (i < flat_keys.size() - 1) {
                result += "|";
            };
        };
        return result;
    };
    EagerDataFrame::EagerDataFrame()
        : table_(arrow::Table::Make(
            arrow::schema({}),
            std::vector<std::shared_ptr<arrow::ChunkedArray>>{},
            0)) {}
    EagerDataFrame::EagerDataFrame(std::shared_ptr<arrow::Table> table)
        : table_(std::move(table)) {
        if (!table_) {
            throw std::invalid_argument("Table pointer cannot be null");
        };
    };
    int64_t EagerDataFrame::num_rows() const noexcept {
        return table_->num_rows();
    };
    int EagerDataFrame::num_cols() const noexcept {
        return table_->num_columns();
    };
    std::shared_ptr<arrow::Schema> EagerDataFrame::schema() const {
        return table_->schema();
    };
    std::vector<std::string> EagerDataFrame::column_names() const {
        std::vector<std::string> names;
        const auto tbl_schema = table_->schema();
        names.reserve(static_cast<size_t>(tbl_schema->num_fields()));
        for (const auto& field : tbl_schema->fields()) {
            names.push_back(field->name());
        };
        return names;
    };
    void EagerDataFrame::print(int64_t max_rows) const {
        const auto safe_rows = max_rows < 0 ? 0 : max_rows;
        const auto limited_rows = std::min<int64_t>(table_->num_rows(), safe_rows);
        const auto preview = table_->Slice(0, limited_rows);
        std::cout << "[" << table_->num_rows() << " rows x " << table_->num_columns() << " cols]" << std::endl;
        std::cout << "Schema:" << std::endl;
        std::cout << table_->schema()->ToString() << std::endl;
        arrow::PrettyPrintOptions options = arrow::PrettyPrintOptions::Defaults();
        options.indent = 2;
        options.window = 0;
        options.null_rep = "null";
        const auto status = arrow::PrettyPrint(*preview, options, &std::cout);
        if (!status.ok()) {
            throw std::runtime_error(status.ToString());
        };
    };
    const std::shared_ptr<arrow::Table>& EagerDataFrame::table() const noexcept {
        return table_;
    };
    EagerDataFrame EagerDataFrame::from_columns(const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& columns) {
        if (columns.empty()) {
            throw std::invalid_argument("Columns vector cannot be empty");
        };
        for (const auto& [name, array] : columns) {
            if (!array) {
                throw std::invalid_argument("Array pointer for column '" + name + "' cannot be null");
            };
        };
        int64_t length = columns[0].second->length();
        for (const auto& [name, array] : columns) {
            if (array->length() != length) {
                throw std::invalid_argument("All columns must have the same length");
            };
        };
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        fields.reserve(columns.size());
        arrays.reserve(columns.size());
        for (const auto& [name, array] : columns) {
            fields.push_back(arrow::field(name, array->type()));
            arrays.push_back(array);
        };
        auto schema = arrow::schema(fields);
        auto table = arrow::Table::Make(schema, arrays);
        return EagerDataFrame(table);
    };
    EagerDataFrame EagerDataFrame::from_columns(const std::map<std::string, std::shared_ptr<arrow::Array>>& columns) {
        std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>> v(columns.begin(), columns.end());
        return from_columns(v);
    };
    void EagerDataFrame::write_csv(const std::string& path) const {
        auto output_result = arrow::io::FileOutputStream::Open(path);
        if (!output_result.ok()) {
            throw std::runtime_error("Failed to open output file: " + output_result.status().ToString());
        };
        auto output = *output_result;
        auto write_options = arrow::csv::WriteOptions::Defaults();
        auto status = arrow::csv::WriteCSV(*table_, write_options, output.get());
        if (!status.ok()) {
            throw std::runtime_error("Failed to write CSV: " + status.ToString());
        };
        auto close_status = output->Close();
        if (!close_status.ok()) {
            throw std::runtime_error("Failed to close CSV output: " + close_status.ToString());
        };
    };
    void EagerDataFrame::write_parquet(const std::string& path, int64_t chunk_size) const {
        auto output_result = arrow::io::FileOutputStream::Open(path);
        if (!output_result.ok()) {
            throw std::runtime_error("Failed to open output file: " + output_result.status().ToString());
        };
        auto output = *output_result;
        auto status = parquet::arrow::WriteTable(
            *table_,
            arrow::default_memory_pool(),
            output,
            chunk_size
        );
        if (!status.ok()) {
            throw std::runtime_error("Failed to write Parquet: " + status.ToString());
        };
        auto close_status = output->Close();
        if (!close_status.ok()) {
            throw std::runtime_error("Failed to close Parquet output: " + close_status.ToString());
        };
    };
    EagerDataFrame read_csv(const std::string& path) {
        const auto input_result = arrow::io::ReadableFile::Open(path);
        if (!input_result.ok()) {
            throw std::runtime_error("Failed to open file: " + input_result.status().ToString());
        };
        const auto input = *std::move(input_result);
        auto read_options = arrow::csv::ReadOptions::Defaults();
        auto parse_options = arrow::csv::ParseOptions::Defaults();
        auto convert_options = arrow::csv::ConvertOptions::Defaults();
        const auto reader_result = arrow::csv::TableReader::Make(
            arrow::io::default_io_context(),
            input,
            read_options,
            parse_options,
            convert_options);
        if (!reader_result.ok()) {
            throw std::runtime_error("Failed to create CSV reader: " + reader_result.status().ToString());
        };
        const auto reader = reader_result.ValueOrDie();
        const auto table_result = reader->Read();
        if (!table_result.ok()) {
            throw std::runtime_error("Failed to read CSV table: " + table_result.status().ToString());
        };
        return EagerDataFrame(table_result.ValueOrDie());
    };
    EagerDataFrame read_parquet(const std::string& path) {
        const auto input_result = arrow::io::ReadableFile::Open(path);
        if (!input_result.ok()) {
            throw std::runtime_error("Failed to open file: " + input_result.status().ToString());
        };
        const auto input = *std::move(input_result);
        auto reader_result = parquet::arrow::OpenFile(
            input,
            arrow::default_memory_pool());
        if (!reader_result.ok()) {
            throw std::runtime_error("Failed to create Parquet reader: " + reader_result.status().ToString());
        };
        auto reader = std::move(reader_result).ValueOrDie();
        std::shared_ptr<arrow::Table> out_table;
        const auto read_status = reader->ReadTable(&out_table);
        if (!read_status.ok()) {
            throw std::runtime_error("Failed to read Parquet table: " + read_status.ToString());
        };
        return EagerDataFrame(out_table);
    };
    EagerDataFrame EagerDataFrame::select(const std::vector<std::string>& columns) const {
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
        fields.reserve(columns.size());
        arrays.reserve(columns.size());
        for (size_t i = 0; i < columns.size(); ++i) {
            const auto& col_name = columns[i];
            auto field = table_->schema()->GetFieldByName(col_name);
            if (!field) {
                throw std::invalid_argument("Column '" + col_name + "' does not exist in the table");
            };
            fields.push_back(field);
            auto array = table_->GetColumnByName(col_name);
            if (!array) {
                throw std::runtime_error("Failed to retrieve column '" + col_name + "' from the table");
            };
            arrays.push_back(array);
        };
        auto new_schema = arrow::schema(fields);
        auto new_table = arrow::Table::Make(new_schema, arrays);
        return EagerDataFrame(new_table);
    };
    EagerDataFrame EagerDataFrame::select_exprs(const std::vector<Expression>& expressions) const {
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
        fields.reserve(expressions.size());
        arrays.reserve(expressions.size());
        for (size_t i = 0; i < expressions.size(); ++i) {
            arrow::Result<arrow::Datum> arr = expressions[i].node()->evaluate(table_);
            if (!arr.ok()) {
                throw std::runtime_error("Failed to evaluate expression '" + expressions[i].node()->to_string() + "': " + arr.status().ToString());
            };
            std::shared_ptr<arrow::ChunkedArray> chunked;
            const auto& datum = arr.ValueOrDie();
            if (datum.kind() == arrow::Datum::SCALAR) {
                chunked = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{
                    dataframelib::broadcast_scalar(datum.scalar(), table_->num_rows())});
            } 
            else if (datum.kind() == arrow::Datum::ARRAY) {
                chunked = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{datum.make_array()});
            } 
            else {
                chunked = datum.chunked_array();
            };
            arrays.push_back(chunked);
            auto type_result = expressions[i].node()->infer_type(*table_->schema());
            if (!type_result.ok()) {
                throw std::runtime_error("Failed to infer type for expression '" + expressions[i].node()->to_string() + "': " + type_result.status().ToString());
            };
            auto type = type_result.ValueOrDie();
            std::string name = "expr_" + std::to_string(i);
            auto node_ptr = expressions[i].node().get();
            if (node_ptr->kind() == ExprKind::Alias) {
                name = static_cast<const AliasExpr*>(node_ptr)->output_name();
            } 
            else if (node_ptr->kind() == ExprKind::Column) {
                name = static_cast<const ColumnExpr*>(node_ptr)->name();
            };
            fields.push_back(arrow::field(name, type));
        };
        auto new_schema = arrow::schema(fields);
        auto new_table = arrow::Table::Make(new_schema, arrays);
        return EagerDataFrame(new_table);
    };
    EagerDataFrame EagerDataFrame::filter(const Expression& predicate) const {
        arrow::Result<arrow::Datum> evaluation = predicate.node()->evaluate(table_);
        if (!evaluation.ok()) {
            throw std::runtime_error("Failed to evaluate filter predicate '" + 
            predicate.node()->to_string() + "': " + evaluation.status().ToString());
        };
        auto value = evaluation.ValueOrDie();
        if (value.type()->id() != arrow::Type::BOOL) {
            throw std::runtime_error("Filter predicate must evaluate to a boolean type, but got " + value.type()->ToString());
        };
        std::shared_ptr<arrow::ChunkedArray> mask;
        if (value.kind() == arrow::Datum::SCALAR) {
            auto scalar_val = std::static_pointer_cast<arrow::BooleanScalar>(value.scalar());
            arrow::BooleanBuilder builder;
            for (int64_t i = 0; i < table_->num_rows(); ++i) DFL_CHECK_OK(builder.Append(scalar_val->value));
            std::shared_ptr<arrow::Array> arr;
            DFL_CHECK_OK(builder.Finish(&arr));
            mask = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{arr});
        }
        else if (value.kind() == arrow::Datum::ARRAY) {
            mask = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{value.make_array()});
        }
        else {
            mask = value.chunked_array();
        };
        std::vector<std::shared_ptr<arrow::Field>> fields = table_->schema()->fields();
        std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
        auto flat_mask = arrow::Concatenate(mask->chunks(), arrow::default_memory_pool()).ValueOrDie();
        auto bool_mask = std::static_pointer_cast<arrow::BooleanArray>(flat_mask);
        for (int i = 0; i < table_->num_columns(); i++) {
            auto type = table_->schema()->field(i)->type();
            std::shared_ptr<arrow::Array> new_chunk;
            std::unique_ptr<arrow::ArrayBuilder> builder;
            DFL_CHECK_OK(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder));
            DFL_CHECK_OK(builder->Reserve(table_->num_rows()));
            filter_column_typed(table_->column(i), *bool_mask, builder.get());
            DFL_CHECK_OK(builder->Finish(&new_chunk));
            arrays.push_back(arrow::ChunkedArray::Make({new_chunk}, type).ValueOrDie());
        };
        auto new_schema = arrow::schema(fields);
        auto new_table = arrow::Table::Make(new_schema, arrays);
        return EagerDataFrame(new_table);
    };
    EagerDataFrame EagerDataFrame::with_column(const std::string& name, const Expression& expr) const {
        arrow::Result<arrow::Datum> evaluation = expr.node()->evaluate(table_);
        if (!evaluation.ok()) {
            throw std::runtime_error("Failed to evaluate expression for new column '" + name + "': " + evaluation.status().ToString());
        };
        auto value = evaluation.ValueOrDie();
        auto type_result = expr.node()->infer_type(*table_->schema());
        if (!type_result.ok()) {
            throw std::runtime_error("Failed to infer type for new column '" 
            + name + "': " + type_result.status().ToString());
        };
        auto field = arrow::field(name, type_result.ValueOrDie());
        std::shared_ptr<arrow::ChunkedArray> new_col;
        if (value.kind() == arrow::Datum::SCALAR) {
            auto flat_arr = dataframelib::broadcast_scalar(value.scalar(), table_->num_rows());
            new_col = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{flat_arr});
        } 
        else {
            if (value.kind() == arrow::Datum::ARRAY) {
                new_col = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{value.make_array()});
            } 
            else {
                new_col = value.chunked_array();
            };
        };
        int index = table_->schema()->GetFieldIndex(name); 
        arrow::Result<std::shared_ptr<arrow::Table>> new_table_result;
        if (index >= 0) {
            new_table_result = table_->SetColumn(index, field, new_col);
        } 
        else {
            new_table_result = table_->AddColumn(table_->num_columns(), field, new_col);
        };
        if (!new_table_result.ok()) {
            throw std::runtime_error("Failed to add new column '" + name + "': " + new_table_result.status().ToString());
        };
        return EagerDataFrame(new_table_result.ValueOrDie());
    };
    EagerDataFrame EagerDataFrame::head(int64_t n) const {
        if (n < 0) {
            throw std::invalid_argument("Number of rows for head() must be non-negative");
        };
        auto new_table = table_->Slice(0, n);
        return EagerDataFrame(new_table);
    };
    EagerDataFrame EagerDataFrame::sort(const std::vector<std::string>& columns, bool ascending) const {
        std::vector<std::shared_ptr<arrow::Array>> sort_arrays;
        for (const auto& col_name : columns) {
            if (!table_->schema()->GetFieldByName(col_name))
                throw std::invalid_argument("Column '" + col_name + "' does not exist");
            auto chunked_col = table_->GetColumnByName(col_name);
            sort_arrays.push_back(arrow::Concatenate(chunked_col->chunks(), arrow::default_memory_pool()).ValueOrDie());
        };
        std::vector<SortKey> keys;
        keys.reserve(sort_arrays.size());
        for (const auto& col : sort_arrays) {
            SortKey k{};
            k.type_id = col->type_id();
            k.array = col.get();
            switch (k.type_id) {
                case arrow::Type::INT64:
                    k.index_64 = static_cast<const arrow::Int64Array*>(col.get())->raw_values();
                    break;
                case arrow::Type::DOUBLE:
                    k.index_double = static_cast<const arrow::DoubleArray*>(col.get())->raw_values();
                    break;
                case arrow::Type::STRING:
                    k.index_string = static_cast<const arrow::StringArray*>(col.get());
                    break;
                default:
                    break;
            };
            keys.push_back(k);
        };
        std::vector<int64_t> indices(table_->num_rows());
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
            for (const auto& k : keys) {
                const bool a_null = k.array->IsNull(a);
                const bool b_null = k.array->IsNull(b);
                if (a_null && b_null) continue;
                if (a_null) return ascending ? false : true;
                if (b_null) return ascending ? true  : false;
                bool less, equal;
                switch (k.type_id) {
                    case arrow::Type::INT64: {
                        int64_t va = k.index_64[a], vb = k.index_64[b];
                        less  = va < vb;
                        equal = va == vb;
                        break;
                    };
                    case arrow::Type::DOUBLE: {
                        double va = k.index_double[a], vb = k.index_double[b];
                        less  = va < vb;
                        equal = va == vb;
                        break;
                    };
                    case arrow::Type::STRING: {
                        auto va = k.index_string->GetView(a);
                        auto vb = k.index_string->GetView(b);
                        less  = va < vb;
                        equal = va == vb;
                        break;
                    };
                    default: {
                        auto sa = k.array->GetScalar(a).ValueOrDie();
                        auto sb = k.array->GetScalar(b).ValueOrDie();
                        if (sa->Equals(*sb)) continue;
                        less = sa->ToString() < sb->ToString();
                        return ascending ? less : !less;
                    };
                };
                if (equal) continue;
                return ascending ? less : !less;
            };
            return false;
        });
        std::vector<std::shared_ptr<arrow::Field>> fields = table_->schema()->fields();
        std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
        for(const auto& chunked_col : table_->columns()) {
            auto type = chunked_col->type();
            auto flat_col = arrow::Concatenate(chunked_col->chunks(), arrow::default_memory_pool()).ValueOrDie();
            std::shared_ptr<arrow::Array> new_chunk;
            std::unique_ptr<arrow::ArrayBuilder> builder;
            DFL_CHECK_OK(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder));
            DFL_CHECK_OK(builder->Reserve(static_cast<int64_t>(indices.size())));
            gather_column_typed(flat_col, indices, builder.get());
            DFL_CHECK_OK(builder->Finish(&new_chunk));
            arrays.push_back(arrow::ChunkedArray::Make({new_chunk}, type).ValueOrDie());
        };
        auto new_schema = arrow::schema(fields);
        auto new_table = arrow::Table::Make(new_schema, arrays);
        return EagerDataFrame(new_table);
    };
    GroupDataFrame EagerDataFrame::group_by(const std::vector<std::string>& keys) const {
        return GroupDataFrame(*this, keys); 
    };
    GroupDataFrame::GroupDataFrame(const EagerDataFrame& df, const std::vector<std::string>& keys) 
                                                        : table_(df.table()), group_keys (keys) {};
    EagerDataFrame GroupDataFrame::agg(const std::vector<Expression>& aggr_exprs) const {
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
        fields.reserve(aggr_exprs.size() + group_keys.size());
        arrays.reserve(aggr_exprs.size() + group_keys.size());
        std::vector<std::shared_ptr<arrow::Array>> flat_keys;
        for (const auto& key : group_keys) {
            auto col = table_->GetColumnByName(key);
            if (!col) {
                throw std::invalid_argument("Group key '" + key + "' does not exist in the table");
            };
            auto flat_col = arrow::Concatenate(col->chunks(), arrow::default_memory_pool()).ValueOrDie();
            flat_keys.push_back(flat_col);
        };
        std::unordered_map<std::string, std::vector<int64_t>> groups;
        std::vector<std::string> grouping_order;
        for (int64_t i = 0; i < table_->num_rows(); ++i) {
            std::string gid = grouping_helper(i, flat_keys);
            auto [it, inserted] = groups.try_emplace(gid);
            if (inserted) {
                grouping_order.push_back(gid);
            };
            it->second.push_back(i);
        };
        for (size_t k = 0; k < group_keys.size(); ++k) {
            auto type = flat_keys[k]->type();
            std::unique_ptr<arrow::ArrayBuilder> builder;
            DFL_CHECK_OK(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder));
            DFL_CHECK_OK(builder->Reserve(static_cast<int64_t>(grouping_order.size())));
            for (const auto& gid : grouping_order) {
                int64_t first_row = groups[gid][0];
                DFL_CHECK_OK(builder->AppendScalar(*flat_keys[k]->GetScalar(first_row).ValueOrDie()));
            };
            std::shared_ptr<arrow::Array> out;
            DFL_CHECK_OK(builder->Finish(&out));
            fields.push_back(arrow::field(group_keys[k], type));
            arrays.push_back(arrow::ChunkedArray::Make({out}, type).ValueOrDie());
        };
        for (size_t i = 0; i < aggr_exprs.size(); ++i) {
            ExprPtr node = aggr_exprs[i].node();
            std::string output_name = "agg_" + std::to_string(i);
            if (node->kind() == ExprKind::Alias) {
                auto alias_node = std::static_pointer_cast<AliasExpr>(node);
                output_name = alias_node->output_name(); 
                node = alias_node->inner(); 
            };
            if (node->kind() != ExprKind::AggFn) {
                throw std::invalid_argument(
                    "agg() expects aggregate expressions (sum/mean/count/min/max), got: "
                    + aggr_exprs[i].node()->to_string());
            };
            auto agg_node = std::static_pointer_cast<AggrExpr>(node);
            auto result = agg_node->child()->evaluate(table_);
            if (!result.ok()) {
                throw std::runtime_error("Failed to evaluate aggregate child: " + result.status().ToString());
            }
            auto datum = result.ValueOrDie();
            std::shared_ptr<arrow::Array> flattened;
            if (datum.kind() == arrow::Datum::SCALAR) {
                flattened = dataframelib::broadcast_scalar(datum.scalar(), table_->num_rows());
            } 
            else if (datum.kind() == arrow::Datum::ARRAY) {
                flattened = datum.make_array();
            } 
            else {
                flattened = arrow::Concatenate(datum.chunked_array()->chunks(), arrow::default_memory_pool()).ValueOrDie();
            };
            arrow::Result<std::shared_ptr<arrow::DataType>> type_result = agg_node->infer_type(*table_->schema());
            if (!type_result.ok()) {
                throw std::runtime_error("Failed to infer type for aggregate child: " + type_result.status().ToString());
            };
            auto type = type_result.ValueOrDie();
            std::unique_ptr<arrow::ArrayBuilder> builder;
                std::shared_ptr<arrow::Array> output_array;
            DFL_CHECK_OK(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder));
            DFL_CHECK_OK(builder->Reserve(static_cast<int64_t>(grouping_order.size())));
            for (size_t j = 0; j < grouping_order.size(); ++j) {
                std::vector<int64_t> rows = groups[grouping_order[j]];
                auto scalar_result = dataframelib::AggregateKernel(agg_node->fn(), flattened, rows);
                DFL_CHECK_OK(builder->AppendScalar(*scalar_result));
            };
            DFL_CHECK_OK(builder->Finish(&output_array));
            fields.push_back(arrow::field(output_name, type));
            arrays.push_back(arrow::ChunkedArray::Make({output_array}, type).ValueOrDie());
        };
        auto final_schema = arrow::schema(fields);
        auto final_table = arrow::Table::Make(final_schema, arrays);
        return EagerDataFrame(final_table);
    };
    EagerDataFrame EagerDataFrame::agg(const std::vector<Expression>& aggr_exprs) const {
        return this->group_by({}).agg(aggr_exprs);
    };
    static std::vector<Expression> pairs_to_exprs(const std::vector<std::pair<std::string, std::string>>& agg_list) {
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
        return exprs;
    };
    EagerDataFrame GroupDataFrame::aggregate(const std::vector<std::pair<std::string, std::string>>& agg_list) const {
        return agg(pairs_to_exprs(agg_list));
    };
    EagerDataFrame EagerDataFrame::aggregate(const std::vector<std::pair<std::string, std::string>>& agg_list) const {
        return this->group_by({}).aggregate(agg_list);
    };
    EagerDataFrame GroupDataFrame::aggregate_map(const std::map<std::string, std::string>& agg_map) const {
        std::vector<std::pair<std::string, std::string>> agg_list(agg_map.begin(), agg_map.end());
        return aggregate(agg_list);
    };
    EagerDataFrame EagerDataFrame::aggregate_map(const std::map<std::string, std::string>& agg_map) const {
        return this->group_by({}).aggregate_map(agg_map);
    };
    EagerDataFrame GroupDataFrame::aggregate(const std::vector<std::pair<std::string, Expression>>& named_aggs) const {
        std::vector<Expression> exprs;
        exprs.reserve(named_aggs.size());
        for (const auto& [name, expr] : named_aggs) {
            exprs.push_back(expr.alias(name));
        }
        return agg(exprs);
    };
    EagerDataFrame EagerDataFrame::aggregate(const std::vector<std::pair<std::string, Expression>>& named_aggs) const {
        return this->group_by({}).aggregate(named_aggs);
    };
    EagerDataFrame EagerDataFrame::join(const EagerDataFrame& other, const std::vector<std::string>& on, const std::string& how) const {
        static const std::unordered_set<std::string> valid_how{"inner", "left", "right", "outer"};
        if (valid_how.find(how) == valid_how.end())
            throw std::invalid_argument("Invalid join type: '" + how + "'");
        if (on.empty())
            throw std::invalid_argument("Join key list cannot be empty");
        for (const auto& keyname : on) {
            auto lf = this->table()->schema()->GetFieldByName(keyname);
            auto rf = other.table()->schema()->GetFieldByName(keyname);
            if (!lf) throw std::invalid_argument("Join key '" + keyname + "' not found in left table");
            if (!rf) throw std::invalid_argument("Join key '" + keyname + "' not found in right table");
            if (!lf->type()->Equals(*rf->type()))
                throw std::invalid_argument("Join key '" + keyname + "' has mismatched types: "
                    + lf->type()->ToString() + " vs " + rf->type()->ToString());
        };
        EagerDataFrame smaller = (this->num_rows() < other.num_rows()) ? *this : other;
        EagerDataFrame larger = (this->num_rows() < other.num_rows()) ? other : *this;
        bool swapped = (this->num_rows() < other.num_rows());
        std::unordered_map<std::string, std::vector<int64_t>> smaller_keys;
        smaller_keys.reserve(static_cast<size_t>(smaller.num_rows()));
        std::vector<std::shared_ptr<arrow::Array>> flat_keys;
        std::vector<int64_t> null_smaller;
        std::vector<std::shared_ptr<arrow::Array>> probe_keys;
        for (const auto& key : on) {
            auto col = smaller.table()->GetColumnByName(key);
            if (!col) throw std::invalid_argument("Join key '" + key + "' not found");
            flat_keys.push_back(
                arrow::Concatenate(col->chunks(), arrow::default_memory_pool()).ValueOrDie());
        };
        for (const auto& key : on) {
            auto col = larger.table()->GetColumnByName(key);
            if (!col) throw std::invalid_argument("Join key '" + key + "' not found");
            probe_keys.push_back(
                arrow::Concatenate(col->chunks(), arrow::default_memory_pool()).ValueOrDie());
        };
        for (int64_t i = 0; i < smaller.num_rows(); ++i) {
            bool any_null = false;
            for (const auto& k : flat_keys) if (k->IsNull(i)) { 
                any_null = true; 
                break; 
            };
            if (any_null) {
                null_smaller.push_back(i);
                continue;
            }
            smaller_keys[grouping_helper(i, flat_keys)].push_back(i);
        };
        std::vector<int64_t> left_indices;
        std::vector<int64_t> right_indices;
        std::set<std::string> matched_keys;
        for (int64_t i = 0; i < larger.num_rows(); ++i) {
            bool any_null = false;
            for (const auto& k : probe_keys) {
                if (k->IsNull(i)) { 
                    any_null = true; 
                    break; 
                };
            };
            std::string probe_key = "";
            bool matched = false;
            if (!any_null) {
                probe_key = grouping_helper(i, probe_keys);
                matched = (smaller_keys.find(probe_key) != smaller_keys.end());
            };
            if (matched) {
                matched_keys.insert(probe_key);
                for (int64_t build_idx : smaller_keys[probe_key]) {
                    if (swapped) {
                        left_indices.push_back(build_idx);
                        right_indices.push_back(i);
                    } 
                    else {
                        left_indices.push_back(i);
                        right_indices.push_back(build_idx);
                    };
                };
            }
            else {
                if (how == "inner") continue;
                else if ((how == "left" || how == "outer") && !swapped) {
                    left_indices.push_back(i);
                    right_indices.push_back(-1);
                }
                else if ((how == "right" || how == "outer") && swapped) {
                    left_indices.push_back(-1);
                    right_indices.push_back(i);
                };
            };
        };
        if (how == "outer" || (how == "left" && swapped) || (how == "right" && !swapped)) {
            for (const auto& k : smaller_keys) {
                if (matched_keys.find(k.first) == matched_keys.end()) {
                    for (int64_t build_idx : k.second) {
                        if (swapped) {
                            left_indices.push_back(build_idx);
                            right_indices.push_back(-1);
                        } 
                        else {
                            left_indices.push_back(-1);
                            right_indices.push_back(build_idx);
                        };
                    };
                };
            };
            for (int64_t null_idx : null_smaller) {
                if (swapped) {
                    left_indices.push_back(null_idx);
                    right_indices.push_back(-1);
                } 
                else {
                    left_indices.push_back(-1);
                    right_indices.push_back(null_idx);
                };
            };
        };
        std::vector<FieldSource> field_sources;
        std::vector<std::shared_ptr<arrow::Field>> final_fields;
        for (const auto& keyname : on) {
            auto field = this->table()->schema()->GetFieldByName(keyname);
            final_fields.push_back(arrow::field(keyname, field->type()));
            field_sources.push_back({Side::Key, keyname});
        };
        for (const auto& field : this->table()->schema()->fields()) {
            if (std::find(on.begin(), on.end(), field->name()) == on.end()) {
                final_fields.push_back(arrow::field(field->name(), field->type()));
                field_sources.push_back({Side::Left, field->name()});
            };
        };
        for (const auto& field : other.table()->schema()->fields()) {
            if (std::find(on.begin(), on.end(), field->name()) != on.end()) continue;
            std::string out_name = field->name();
            if (this->table()->schema()->GetFieldByName(field->name()) != nullptr) {
                out_name = field->name() + "_right";
            };
            final_fields.push_back(arrow::field(out_name, field->type()));
            field_sources.push_back({Side::Right, field->name()});
        };
        std::vector<std::shared_ptr<arrow::Array>> flattened_left;
        for (int i = 0; i < this->table()->num_columns(); i++) {
            flattened_left.push_back(arrow::Concatenate(
                this->table()->column(i)->chunks(), arrow::default_memory_pool()).ValueOrDie());
        };
        std::vector<std::shared_ptr<arrow::Array>> flattened_right;
        for (int i = 0; i < other.table()->num_columns(); i++) {
            flattened_right.push_back(arrow::Concatenate(
                other.table()->column(i)->chunks(), arrow::default_memory_pool()).ValueOrDie());
        };
        std::vector<std::shared_ptr<arrow::ChunkedArray>> final_arrays;
        for (size_t f = 0; f < final_fields.size(); ++f) {
            const auto& source = field_sources[f];
            auto type = final_fields[f]->type();
            std::unique_ptr<arrow::ArrayBuilder> builder;
            DFL_CHECK_OK(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder));
            DFL_CHECK_OK(builder->Reserve(static_cast<int64_t>(left_indices.size())));
            if (source.side == Side::Key) {
                auto left_col = flattened_left[this->table()->schema()->GetFieldIndex(source.src_name)];
                auto right_col = flattened_right[other.table()->schema()->GetFieldIndex(source.src_name)];
                for (size_t i = 0; i < left_indices.size(); ++i) {
                    if (left_indices[i] != -1) {
                        DFL_CHECK_OK(builder->AppendScalar(*left_col->GetScalar(left_indices[i]).ValueOrDie()));
                    }
                    else if (right_indices[i] != -1) {
                        DFL_CHECK_OK(builder->AppendScalar(*right_col->GetScalar(right_indices[i]).ValueOrDie()));
                    }
                    else {
                        DFL_CHECK_OK(builder->AppendNull());
                    };
                };
            }
            else if (source.side == Side::Left) {
                auto col = flattened_left[this->table()->schema()->GetFieldIndex(source.src_name)];
                gather_column_typed(col, left_indices, builder.get());
            }
            else {
                auto col = flattened_right[other.table()->schema()->GetFieldIndex(source.src_name)];
                gather_column_typed(col, right_indices, builder.get());
            };
            std::shared_ptr<arrow::Array> out;
            DFL_CHECK_OK(builder->Finish(&out));
            final_arrays.push_back(arrow::ChunkedArray::Make({out}, type).ValueOrDie());
        };
        std::unordered_set<std::string> seen;
        for (const auto& f : final_fields) {
            if (!seen.insert(f->name()).second)
                throw std::invalid_argument(
                    "join produced duplicate output column: " + f->name());
        };
        auto new_schema = arrow::schema(final_fields);
        auto new_table = arrow::Table::Make(new_schema, final_arrays);
        return EagerDataFrame(new_table);
    };
};