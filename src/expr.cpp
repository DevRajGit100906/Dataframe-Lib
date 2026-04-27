#include "dataframelib/expr.h"
#include <utility>
#include <arrow/type.h>
#include <arrow/scalar.h>
#include <arrow/status.h>
#include "dataframelib/kernel.h"
#include <arrow/array/concatenate.h>
namespace dataframelib {
    arrow::Result<std::shared_ptr<arrow::DataType>> promote_numeric(
        const std::shared_ptr<arrow::DataType>& l, const std::shared_ptr<arrow::DataType>& r) {
        if (l->id() == arrow::Type::NA && r->id() == arrow::Type::NA) {
            return arrow::null();
        };
        if (l->id() == arrow::Type::NA) {
            if (!arrow::is_numeric(r->id())) {
                return arrow::Status::TypeError("non-numeric operand in arithmetic");
            };
            return arrow::is_floating(r->id()) ? arrow::float64() : arrow::int64();
        };
        if (r->id() == arrow::Type::NA) {
            if (!arrow::is_numeric(l->id())) {
                return arrow::Status::TypeError("non-numeric operand in arithmetic");
            };
            return arrow::is_floating(l->id()) ? arrow::float64() : arrow::int64();
        };
        if (!arrow::is_numeric(l->id()) || !arrow::is_numeric(r->id())) {
            return arrow::Status::TypeError("non-numeric operand in arithmetic");
        };
        if (arrow::is_floating(l->id()) || arrow::is_floating(r->id())) {
            return arrow::float64();
        };
        return arrow::int64();
    };
    ColumnExpr::ColumnExpr(std::string name)
                : name_(std::move(name)) {};
    ExprKind ColumnExpr::kind() const {
        return ExprKind::Column;
    };
    arrow::Result<arrow::Datum> ColumnExpr::evaluate(
        const std::shared_ptr<arrow::Table>& table) const {
        const auto column = table->GetColumnByName(name_);
        if (!column) {
            return arrow::Status::KeyError("Column not found: " + name_);
        };
        std::shared_ptr<arrow::Array> flat_array;
        if (column->num_chunks() == 1) {
            flat_array = column->chunk(0);
        }
        else {
            auto concat_result = arrow::Concatenate(column->chunks());
            if (!concat_result.ok()) return concat_result.status();
            flat_array = concat_result.ValueOrDie();
        };
        return arrow::Datum(flat_array);
    };
    arrow::Result<std::shared_ptr<arrow::DataType>> ColumnExpr::infer_type(
                                    const arrow::Schema& schema) const {
        const auto field = schema.GetFieldByName(name_);
        if (!field) {
            return arrow::Status::KeyError("Column not found in schema: " + name_);
        };
        return field->type();
    };
    std::string ColumnExpr::to_string() const {
        return name_;
    };
    const std::string& ColumnExpr::name() const {
        return name_;
    };
    LiteralExpr::LiteralExpr(std::shared_ptr<arrow::Scalar> v)
                                    : value_(std::move(v)) {};
    ExprKind LiteralExpr::kind() const {
        return ExprKind::Literal;
    };
    arrow::Result<arrow::Datum> LiteralExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
        return arrow::Datum(value_);
    };
    arrow::Result<std::shared_ptr<arrow::DataType>> LiteralExpr::infer_type(
                                    const arrow::Schema& schema) const {
        return value_->type;
    };
    std::string LiteralExpr::to_string() const {
        return value_->ToString();
    };
    const std::shared_ptr<arrow::Scalar>& LiteralExpr::value() const {
        return value_;
    };
    AliasExpr::AliasExpr(ExprPtr inner, std::string name)
    : inner_(std::move(inner)), name_(std::move(name)) {};
    ExprKind AliasExpr::kind() const {
        return ExprKind::Alias;
    };
    arrow::Result<arrow::Datum> AliasExpr::evaluate(
        const std::shared_ptr<arrow::Table>& table) const {
        return inner_->evaluate(table);
    };
    arrow::Result<std::shared_ptr<arrow::DataType>> AliasExpr::infer_type(
                                    const arrow::Schema& schema) const {
        return inner_->infer_type(schema);
    };
    std::string AliasExpr::to_string() const {
        return inner_->to_string() + " AS " + name_;
    };
    std::vector<ExprPtr> AliasExpr::children() const {
        return {inner_};
    };
    const std::string& AliasExpr::output_name() const {
        return name_;
    };
    const ExprPtr& AliasExpr::inner() const {
        return inner_;
    };
    BinaryOpExpr::BinaryOpExpr(BinOp op, ExprPtr lhs, ExprPtr rhs)
        : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {};
    ExprKind BinaryOpExpr::kind() const {
        return ExprKind::BinaryOp;
    };
    BinOp BinaryOpExpr::op() const {
        return op_;
    };
    arrow::Result<arrow::Datum> BinaryOpExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
        auto result_lhs = lhs_->evaluate(table);
        if (!result_lhs.ok()) {
            return result_lhs.status();
        };
        auto result_rhs = rhs_->evaluate(table);
        if (!result_rhs.ok()) {
            return result_rhs.status();
        };
        const auto& lhs_datum = result_lhs.ValueOrDie();
        const auto& rhs_datum = result_rhs.ValueOrDie();
        std::shared_ptr<arrow::Array> left_array;
        std::shared_ptr<arrow::Array> right_array;
        const int64_t target_len = table->num_rows();
        if (lhs_datum.is_scalar()) {
            left_array = dataframelib::broadcast_scalar(lhs_datum.scalar(), target_len);
        } 
        else {
            left_array = lhs_datum.make_array(); 
        };
        if (rhs_datum.is_scalar()) {
            right_array = dataframelib::broadcast_scalar(rhs_datum.scalar(), target_len);
        } 
        else {
            right_array = rhs_datum.make_array();
        };
        switch (op_) {
            case BinOp::Add: 
                return arrow::Datum(dataframelib::add(left_array, right_array));
            case BinOp::Sub: 
                return arrow::Datum(dataframelib::sub(left_array, right_array));
            case BinOp::Mul: 
                return arrow::Datum(dataframelib::mul(left_array, right_array));
            case BinOp::Div:
                return arrow::Datum(dataframelib::div(left_array, right_array));
            case BinOp::Mod:
                return arrow::Datum(dataframelib::mod(left_array, right_array));
            case BinOp::Eq:
                return arrow::Datum(dataframelib::eq(left_array, right_array));
            case BinOp::Ne:
                return arrow::Datum(dataframelib::ne(left_array, right_array));
            case BinOp::Lt:
                return arrow::Datum(dataframelib::lt(left_array, right_array));
            case BinOp::Le:
                return arrow::Datum(dataframelib::le(left_array, right_array));
            case BinOp::Gt:
                return arrow::Datum(dataframelib::gt(left_array, right_array));
            case BinOp::Ge:
                return arrow::Datum(dataframelib::ge(left_array, right_array));
            case BinOp::And:
                return arrow::Datum(dataframelib::and_kleene(left_array, right_array));
            case BinOp::Or:
                return arrow::Datum(dataframelib::or_kleene(left_array, right_array));            
            default:
                return arrow::Status::Invalid("Unknown binary operator");
        };
    };
    arrow::Result<std::shared_ptr<arrow::DataType>> BinaryOpExpr::infer_type(
                                    const arrow::Schema& schema) const {
        if (is_comparison() || is_boolean()) {
            return arrow::boolean();
        };
        auto result_lhs = lhs_->infer_type(schema);
        if (!result_lhs.ok()) {
            return result_lhs.status();
        };
        auto result_rhs = rhs_->infer_type(schema);
        if (!result_rhs.ok()) {
            return result_rhs.status();
        };
        return promote_numeric(result_lhs.ValueOrDie(), result_rhs.ValueOrDie());
    };
    std::vector<ExprPtr> BinaryOpExpr::children() const {
        return {lhs_, rhs_};
    };
    std::string BinaryOpExpr::to_string() const {
        const char* sym = "?";
        switch (op_) {
            case BinOp::Add: sym = "+"; break;
            case BinOp::Sub: sym = "-"; break;
            case BinOp::Mul: sym = "*"; break;
            case BinOp::Div: sym = "/"; break;
            case BinOp::Mod: sym = "%"; break;
            case BinOp::Eq:  sym = "=="; break;
            case BinOp::Ne:  sym = "!="; break;
            case BinOp::Lt:  sym = "<"; break;
            case BinOp::Le:  sym = "<="; break;
            case BinOp::Gt:  sym = ">"; break;
            case BinOp::Ge:  sym = ">="; break;
            case BinOp::And: sym = "AND"; break;
            case BinOp::Or:  sym = "OR"; break;
        };
        return lhs_->to_string() + " " + sym + " " + rhs_->to_string();
    };
    UnaryOpExpr::UnaryOpExpr(UnaryOp op, ExprPtr child)
                : op_(op), child_(std::move(child)) {};
    ExprKind UnaryOpExpr::kind() const {
        return ExprKind::UnaryOp;
    };
    arrow::Result<arrow::Datum> UnaryOpExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
        auto result_child = child_->evaluate(table);
        if (!result_child.ok()) return result_child.status();
        const auto child_datum = result_child.ValueOrDie();
        const int64_t target_len = table->num_rows();
        std::shared_ptr<arrow::Array> input_array;
        if (child_datum.is_scalar()) {
            input_array = dataframelib::broadcast_scalar(child_datum.scalar(), target_len);
        } 
        else {
            input_array = child_datum.make_array();
        };
        switch (op_) {
            case UnaryOp::Abs:
                return arrow::Datum(dataframelib::abs(input_array));
            case UnaryOp::IsNull:
                return arrow::Datum(dataframelib::is_null(input_array));
            case UnaryOp::IsNotNull:
                return arrow::Datum(dataframelib::is_not_null(input_array));
            case UnaryOp::Not:
                return arrow::Datum(dataframelib::not_kleene(input_array));
            case UnaryOp::Negate:
                return arrow::Datum(dataframelib::negate(input_array));
            default:
                return arrow::Status::Invalid("Unknown unary operator");
        };
    };
    arrow::Result<std::shared_ptr<arrow::DataType>> UnaryOpExpr::infer_type(
                                    const arrow::Schema& schema) const {
        if (op_ == UnaryOp::IsNull || op_ == UnaryOp::IsNotNull || op_ == UnaryOp::Not) {
            return arrow::boolean();
        };                                
        auto result_child = child_->infer_type(schema);
        if (!result_child.ok()) {
            return result_child.status();
        };
        return result_child;
    };
    std::vector<ExprPtr> UnaryOpExpr::children() const {
        return {child_};
    };
    std::string UnaryOpExpr::to_string() const {
        const char* name = "?";
        switch (op_) {
            case UnaryOp::Abs:       name = "abs"; break;
            case UnaryOp::IsNull:    name = "is_null"; break;
            case UnaryOp::IsNotNull: name = "is_not_null"; break;
            case UnaryOp::Not:       name = "not"; break;
            case UnaryOp::Negate:    name = "neg"; break;
        };
        return std::string(name) + "(" + child_->to_string() + ")";
    };
    StringFnExpr::StringFnExpr(StringFn fn, ExprPtr child, std::string arg)
                : fn_(fn), child_(std::move(child)), arg_(std::move(arg)) {};
    ExprKind StringFnExpr::kind() const {
        return ExprKind::StringFn;
    };
    arrow::Result<arrow::Datum> StringFnExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
        auto result_child = child_->evaluate(table);
        if (!result_child.ok()) return result_child.status();
        const auto child_datum = result_child.ValueOrDie();
        const int64_t target_len = table->num_rows();
        std::shared_ptr<arrow::Array> input_array;
        if (child_datum.is_scalar()) {
            input_array = dataframelib::broadcast_scalar(child_datum.scalar(), target_len);
        } 
        else {
            input_array = child_datum.make_array();
        };
        switch (fn_) {
            case StringFn::Length:
                return arrow::Datum(dataframelib::utf8_length(input_array));
            case StringFn::ToLower:
                return arrow::Datum(dataframelib::to_lower(input_array));
            case StringFn::ToUpper:
                return arrow::Datum(dataframelib::to_upper(input_array));
            case StringFn::Contains:
            return arrow::Datum(dataframelib::contains(input_array, arg_));
            case StringFn::StartsWith:
                return arrow::Datum(dataframelib::starts_with(input_array, arg_));
            case StringFn::EndsWith:
                return arrow::Datum(dataframelib::ends_with(input_array, arg_));
            default:
                return arrow::Status::Invalid("Unknown string function");
        };
    };
    arrow::Result<std::shared_ptr<arrow::DataType>> StringFnExpr::infer_type(
                                    const arrow::Schema& schema) const {
        switch (fn_) {
            case StringFn::Length: return arrow::int64();
            case StringFn::ToLower:
            case StringFn::ToUpper:
                return arrow::utf8(); 
            case StringFn::Contains:
            case StringFn::StartsWith:
            case StringFn::EndsWith:
                return arrow::boolean();
        };
        return arrow::Status::Invalid("Unknown string function");
    };
    std::vector<ExprPtr> StringFnExpr::children() const {
        return {child_};
    };
    std::string StringFnExpr::to_string() const {
        const char* name = "?";
        switch (fn_) {
            case StringFn::Length:       name = "length"; break;
            case StringFn::ToLower:      name = "lower"; break;
            case StringFn::ToUpper:      name = "upper"; break;
            case StringFn::Contains:     name = "contains"; break;
            case StringFn::StartsWith:   name = "starts_with"; break;
            case StringFn::EndsWith:     name = "ends_with"; break;
        };
        return std::string(name) + "(" + child_->to_string() + ")";
    };
    AggrExpr::AggrExpr(AggFn fn, ExprPtr child)
                : fn_(fn), child_(std::move(child)) {};
    ExprKind AggrExpr::kind() const {
        return ExprKind::AggFn;
    };
    arrow::Result<arrow::Datum> AggrExpr::evaluate(
                                    const std::shared_ptr<arrow::Table>&) const {
        return arrow::Status::Invalid(
            "AggrExpr cannot be evaluated outside of agg()");
    };
    arrow::Result<std::shared_ptr<arrow::DataType>> AggrExpr::infer_type(
                                    const arrow::Schema& schema) const {
        switch (fn_) {
            case AggFn::Count:
                return arrow::int64();
            case AggFn::Sum:
            case AggFn::Mean:
            case AggFn::Min:
            case AggFn::Max: {
                auto child_type_result = child_->infer_type(schema);
                if (!child_type_result.ok()) {
                    return child_type_result.status();
                };
                const auto child_type = child_type_result.ValueOrDie();
                if (!arrow::is_numeric(child_type->id())) {
                    return arrow::Status::TypeError("Aggregate function "
                            + to_string() + " requires numeric input");
                };
                return child_type;
            };
            default:
                return arrow::Status::Invalid("Unknown aggregate function");
        };
    };
    AggFn AggrExpr::fn() const {
        return fn_;
    };
    const ExprPtr& AggrExpr::child() const {
        return child_;
    };
    std::vector<ExprPtr> AggrExpr::children() const {
        return {child_};
    };
    std::string AggrExpr::to_string() const {
        const char* name = "?";
        switch (fn_) {
            case AggFn::Sum:   name = "SUM"; break;
            case AggFn::Mean:  name = "MEAN"; break;
            case AggFn::Count: name = "COUNT"; break;
            case AggFn::Min:   name = "MIN"; break;
            case AggFn::Max:   name = "MAX"; break;
        };
        return std::string(name) + "(" + child_->to_string() + ")";
    };
    Expression::Expression(ExprPtr e) : expr_(std::move(e)) {};
    const ExprPtr& Expression::node() const {
        return expr_;
    };
    Expression Expression::alias(std::string name) const {
        return Expression(std::make_shared<AliasExpr>(node(), std::move(name)));
    };
    Expression Expression::abs() const {
        return Expression(std::make_shared<UnaryOpExpr>(UnaryOp::Abs, node()));
    };
    Expression Expression::is_null() const {
        return Expression(std::make_shared<UnaryOpExpr>(UnaryOp::IsNull, node()));
    };
    Expression Expression::is_not_null() const {
        return Expression(std::make_shared<UnaryOpExpr>(UnaryOp::IsNotNull, node()));
    };
    Expression Expression::length() const {
        return Expression(std::make_shared<StringFnExpr>(StringFn::Length, node()));
    };
    Expression Expression::contains(std::string s) const {
        return Expression(std::make_shared<StringFnExpr>(StringFn::Contains, node(), std::move(s)));
    };
    Expression Expression::starts_with(std::string s) const {
        return Expression(std::make_shared<StringFnExpr>(StringFn::StartsWith, node(), std::move(s)));
    };
    Expression Expression::ends_with(std::string s) const {
        return Expression(std::make_shared<StringFnExpr>(StringFn::EndsWith, node(), std::move(s)));
    };
    Expression Expression::to_lower() const {
        return Expression(std::make_shared<StringFnExpr>(StringFn::ToLower, node()));
    };
    Expression Expression::to_upper() const {
        return Expression(std::make_shared<StringFnExpr>(StringFn::ToUpper, node()));
    };
    Expression Expression::sum() const {
        return Expression(std::make_shared<AggrExpr>(AggFn::Sum, node()));
    };
    Expression Expression::mean() const {
        return Expression(std::make_shared<AggrExpr>(AggFn::Mean, node()));
    };
    Expression Expression::count() const {
        return Expression(std::make_shared<AggrExpr>(AggFn::Count, node()));
    };
    Expression Expression::min() const {
        return Expression(std::make_shared<AggrExpr>(AggFn::Min, node()));
    };
    Expression Expression::max() const {
        return Expression(std::make_shared<AggrExpr>(AggFn::Max, node()));
    };
}