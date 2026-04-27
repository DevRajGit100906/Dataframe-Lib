#pragma once
#include <arrow/api.h>
#include <arrow/scalar.h>
#include <memory>
#include <vector>
#include <string>
#include <utility>
namespace dataframelib {
    class Expr;
    using ExprPtr = std::shared_ptr<Expr>;
    enum class ExprKind {
        Column, Literal, Alias,
        BinaryOp, UnaryOp, StringFn, AggFn
    };
    enum class BinOp {
        Add, Sub, Mul, Div, Mod,
        Eq, Ne, Lt, Le, Gt, Ge,
        And, Or
    };
    enum class UnaryOp { 
        Abs, 
        IsNull, IsNotNull, 
        Not, Negate 
    };
    enum class StringFn { 
        Length, 
        ToLower, ToUpper, 
        Contains, StartsWith, EndsWith 
    };
    enum class AggFn { 
        Sum, Mean, 
        Count, 
        Min, Max 
    };
    arrow::Result<std::shared_ptr<arrow::DataType>> promote_numeric(
        const std::shared_ptr<arrow::DataType>& l,
        const std::shared_ptr<arrow::DataType>& r);
    class Expr {
        public:
        virtual ~Expr() = default;
        virtual ExprKind kind() const = 0;
        virtual arrow::Result<arrow::Datum> evaluate(
            const std::shared_ptr<arrow::Table>& table) const = 0;
        virtual arrow::Result<std::shared_ptr<arrow::DataType>> infer_type(
            const arrow::Schema& schema) const = 0;
        virtual std::string to_string() const = 0;
        virtual std::vector<ExprPtr> children() const { return {}; }
    };
    class ColumnExpr : public Expr {
        public:
            explicit ColumnExpr(std::string name);
            ExprKind kind() const override;
            arrow::Result<arrow::Datum> evaluate(
                const std::shared_ptr<arrow::Table>& table) const override;
            arrow::Result<std::shared_ptr<arrow::DataType>> infer_type(
                const arrow::Schema& schema) const override;
            std::string to_string() const override;
            const std::string& name() const;
        private:
            std::string name_;
    };
    class LiteralExpr : public Expr {
    public:
        explicit LiteralExpr(std::shared_ptr<arrow::Scalar> v);
        ExprKind kind() const override;
        arrow::Result<arrow::Datum> evaluate(
            const std::shared_ptr<arrow::Table>& table) const override;
        arrow::Result<std::shared_ptr<arrow::DataType>> infer_type(
            const arrow::Schema& schema) const override;
        std::string to_string() const override;
        const std::shared_ptr<arrow::Scalar>& value() const;
    private:
        std::shared_ptr<arrow::Scalar> value_;
    };
    class AliasExpr : public Expr {
        public:
            AliasExpr(ExprPtr inner, std::string name);
            ExprKind kind() const override;
            arrow::Result<arrow::Datum> evaluate(
                const std::shared_ptr<arrow::Table>& t) const override;
            arrow::Result<std::shared_ptr<arrow::DataType>> infer_type(
                const arrow::Schema& s) const override;
            std::string to_string() const override;
            std::vector<ExprPtr> children() const override;
            const std::string& output_name() const;
            const ExprPtr& inner() const;
        private:
            ExprPtr inner_;
            std::string name_;
    };
    class BinaryOpExpr : public Expr {
        public:
            BinaryOpExpr(BinOp op, ExprPtr lhs, ExprPtr rhs);
            ExprKind kind() const override;
            BinOp op() const;
            arrow::Result<arrow::Datum> evaluate(
                const std::shared_ptr<arrow::Table>& t) const override;
            arrow::Result<std::shared_ptr<arrow::DataType>> infer_type(
                const arrow::Schema& s) const override;
            std::vector<ExprPtr> children() const override;
            std::string to_string() const override;
        private:
            bool is_comparison() const {
                return op_ >= BinOp::Eq && op_ <= BinOp::Ge;
            };
            bool is_boolean() const { 
                return op_ == BinOp::And || op_ == BinOp::Or; 
            };
            BinOp op_;
            ExprPtr lhs_, rhs_;
    };
    class UnaryOpExpr : public Expr {
        public:
            UnaryOpExpr(UnaryOp op, ExprPtr child);
            ExprKind kind() const override;
            UnaryOp op() const { return op_; }
            arrow::Result<arrow::Datum> evaluate(
                const std::shared_ptr<arrow::Table>& t) const override;
            arrow::Result<std::shared_ptr<arrow::DataType>> infer_type(
                const arrow::Schema& s) const override;
            std::vector<ExprPtr> children() const override;
            std::string to_string() const override;
        private:
            UnaryOp op_;
            ExprPtr child_;
    };
    class StringFnExpr : public Expr {
        public:
            StringFnExpr(StringFn fn, ExprPtr child, std::string arg = "");
            ExprKind kind() const override;
            StringFn fn() const { return fn_; }
            const std::string& arg() const { return arg_; }
            arrow::Result<arrow::Datum> evaluate(
                const std::shared_ptr<arrow::Table>& t) const override;
            arrow::Result<std::shared_ptr<arrow::DataType>> infer_type(
                const arrow::Schema& s) const override;
            std::vector<ExprPtr> children() const override;
            std::string to_string() const override;
        private:
            StringFn fn_;
            ExprPtr child_;
            std::string arg_;
    };
    class AggrExpr : public Expr {
        public:
            AggrExpr(AggFn fn, ExprPtr child);
            ExprKind kind() const override;
            arrow::Result<arrow::Datum> evaluate(
                const std::shared_ptr<arrow::Table>& t) const override;
            arrow::Result<std::shared_ptr<arrow::DataType>> infer_type(
                const arrow::Schema& s) const override;
            AggFn fn() const;
            const ExprPtr& child() const;
            std::vector<ExprPtr> children() const override;
            std::string to_string() const override;
        private:            
            AggFn fn_;
            ExprPtr child_;
    };
    class Expression {
        public:
            Expression() = default;
            explicit Expression(ExprPtr e);
            const ExprPtr& node() const;
            Expression alias(std::string name) const;
            Expression abs() const;
            Expression is_null() const;
            Expression is_not_null() const;
            Expression length() const;
            Expression contains(std::string s) const;
            Expression starts_with(std::string s) const;
            Expression ends_with(std::string s) const;
            Expression to_lower() const;
            Expression to_upper() const;
            Expression sum() const;
            Expression mean() const;
            Expression count() const;
            Expression min() const;
            Expression max() const;
        private:
            ExprPtr expr_;
    };
    inline Expression col(std::string name) {return Expression(std::make_shared<ColumnExpr>(std::move(name)));};
    inline Expression lit(int64_t v) {return Expression(std::make_shared<LiteralExpr>(std::make_shared<arrow::Int64Scalar>(v)));};
    inline Expression lit(int v) { return lit(static_cast<int64_t>(v)); }
    inline Expression lit(double v) {return Expression(std::make_shared<LiteralExpr>(std::make_shared<arrow::DoubleScalar>(v)));};
    inline Expression lit(bool v) {return Expression(std::make_shared<LiteralExpr>(std::make_shared<arrow::BooleanScalar>(v)));};
    inline Expression lit(const std::string& v) {return Expression(std::make_shared<LiteralExpr>(std::make_shared<arrow::StringScalar>(v)));};
    inline Expression operator+(const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Add, a.node(), b.node()));};
    inline Expression operator-(const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Sub, a.node(), b.node()));};
    inline Expression operator*(const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Mul, a.node(), b.node()));};
    inline Expression operator/(const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Div, a.node(), b.node()));};
    inline Expression operator%(const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Mod, a.node(), b.node()));};
    inline Expression operator==(const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Eq, a.node(), b.node()));};
    inline Expression operator!=(const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Ne, a.node(), b.node()));};
    inline Expression operator< (const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Lt, a.node(), b.node()));};
    inline Expression operator<=(const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Le, a.node(), b.node()));};
    inline Expression operator> (const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Gt, a.node(), b.node()));};
    inline Expression operator>=(const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Ge, a.node(), b.node()));};
    inline Expression operator& (const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::And, a.node(), b.node()));};
    inline Expression operator| (const Expression& a, const Expression& b) {return Expression(std::make_shared<BinaryOpExpr>(BinOp::Or, a.node(), b.node()));};
    inline Expression operator~ (const Expression& a) {return Expression(std::make_shared<UnaryOpExpr>(UnaryOp::Not, a.node()));};

    // ---- mixed Expression op primitive overloads (so col("x") + 5 works) ----
    inline Expression operator+(const Expression& a, int b) {return a + lit(b);};
    inline Expression operator+(const Expression& a, int64_t b) {return a + lit(b);};
    inline Expression operator+(const Expression& a, double b) {return a + lit(b);};
    inline Expression operator+(int a, const Expression& b) {return lit(a) + b;};
    inline Expression operator+(int64_t a, const Expression& b) {return lit(a) + b;};
    inline Expression operator+(double a, const Expression& b) {return lit(a) + b;};

    inline Expression operator-(const Expression& a, int b) {return a - lit(b);};
    inline Expression operator-(const Expression& a, int64_t b) {return a - lit(b);};
    inline Expression operator-(const Expression& a, double b) {return a - lit(b);};
    inline Expression operator-(int a, const Expression& b) {return lit(a) - b;};
    inline Expression operator-(int64_t a, const Expression& b) {return lit(a) - b;};
    inline Expression operator-(double a, const Expression& b) {return lit(a) - b;};

    inline Expression operator*(const Expression& a, int b) {return a * lit(b);};
    inline Expression operator*(const Expression& a, int64_t b) {return a * lit(b);};
    inline Expression operator*(const Expression& a, double b) {return a * lit(b);};
    inline Expression operator*(int a, const Expression& b) {return lit(a) * b;};
    inline Expression operator*(int64_t a, const Expression& b) {return lit(a) * b;};
    inline Expression operator*(double a, const Expression& b) {return lit(a) * b;};

    inline Expression operator/(const Expression& a, int b) {return a / lit(b);};
    inline Expression operator/(const Expression& a, int64_t b) {return a / lit(b);};
    inline Expression operator/(const Expression& a, double b) {return a / lit(b);};
    inline Expression operator/(int a, const Expression& b) {return lit(a) / b;};
    inline Expression operator/(int64_t a, const Expression& b) {return lit(a) / b;};
    inline Expression operator/(double a, const Expression& b) {return lit(a) / b;};

    inline Expression operator%(const Expression& a, int b) {return a % lit(b);};
    inline Expression operator%(const Expression& a, int64_t b) {return a % lit(b);};
    inline Expression operator%(int a, const Expression& b) {return lit(a) % b;};
    inline Expression operator%(int64_t a, const Expression& b) {return lit(a) % b;};

    inline Expression operator==(const Expression& a, int b) {return a == lit(b);};
    inline Expression operator==(const Expression& a, int64_t b) {return a == lit(b);};
    inline Expression operator==(const Expression& a, double b) {return a == lit(b);};
    inline Expression operator==(const Expression& a, const std::string& b) {return a == lit(b);};
    inline Expression operator==(const Expression& a, const char* b) {return a == lit(std::string(b));};
    inline Expression operator==(int a, const Expression& b) {return lit(a) == b;};
    inline Expression operator==(int64_t a, const Expression& b) {return lit(a) == b;};
    inline Expression operator==(double a, const Expression& b) {return lit(a) == b;};
    inline Expression operator==(const std::string& a, const Expression& b) {return lit(a) == b;};
    inline Expression operator==(const char* a, const Expression& b) {return lit(std::string(a)) == b;};

    inline Expression operator!=(const Expression& a, int b) {return a != lit(b);};
    inline Expression operator!=(const Expression& a, int64_t b) {return a != lit(b);};
    inline Expression operator!=(const Expression& a, double b) {return a != lit(b);};
    inline Expression operator!=(const Expression& a, const std::string& b) {return a != lit(b);};
    inline Expression operator!=(const Expression& a, const char* b) {return a != lit(std::string(b));};
    inline Expression operator!=(int a, const Expression& b) {return lit(a) != b;};
    inline Expression operator!=(int64_t a, const Expression& b) {return lit(a) != b;};
    inline Expression operator!=(double a, const Expression& b) {return lit(a) != b;};
    inline Expression operator!=(const std::string& a, const Expression& b) {return lit(a) != b;};
    inline Expression operator!=(const char* a, const Expression& b) {return lit(std::string(a)) != b;};

    inline Expression operator<(const Expression& a, int b) {return a < lit(b);};
    inline Expression operator<(const Expression& a, int64_t b) {return a < lit(b);};
    inline Expression operator<(const Expression& a, double b) {return a < lit(b);};
    inline Expression operator<(const Expression& a, const std::string& b) {return a < lit(b);};
    inline Expression operator<(int a, const Expression& b) {return lit(a) < b;};
    inline Expression operator<(int64_t a, const Expression& b) {return lit(a) < b;};
    inline Expression operator<(double a, const Expression& b) {return lit(a) < b;};

    inline Expression operator<=(const Expression& a, int b) {return a <= lit(b);};
    inline Expression operator<=(const Expression& a, int64_t b) {return a <= lit(b);};
    inline Expression operator<=(const Expression& a, double b) {return a <= lit(b);};
    inline Expression operator<=(const Expression& a, const std::string& b) {return a <= lit(b);};
    inline Expression operator<=(int a, const Expression& b) {return lit(a) <= b;};
    inline Expression operator<=(int64_t a, const Expression& b) {return lit(a) <= b;};
    inline Expression operator<=(double a, const Expression& b) {return lit(a) <= b;};

    inline Expression operator>(const Expression& a, int b) {return a > lit(b);};
    inline Expression operator>(const Expression& a, int64_t b) {return a > lit(b);};
    inline Expression operator>(const Expression& a, double b) {return a > lit(b);};
    inline Expression operator>(const Expression& a, const std::string& b) {return a > lit(b);};
    inline Expression operator>(int a, const Expression& b) {return lit(a) > b;};
    inline Expression operator>(int64_t a, const Expression& b) {return lit(a) > b;};
    inline Expression operator>(double a, const Expression& b) {return lit(a) > b;};

    inline Expression operator>=(const Expression& a, int b) {return a >= lit(b);};
    inline Expression operator>=(const Expression& a, int64_t b) {return a >= lit(b);};
    inline Expression operator>=(const Expression& a, double b) {return a >= lit(b);};
    inline Expression operator>=(const Expression& a, const std::string& b) {return a >= lit(b);};
    inline Expression operator>=(int a, const Expression& b) {return lit(a) >= b;};
    inline Expression operator>=(int64_t a, const Expression& b) {return lit(a) >= b;};
    inline Expression operator>=(double a, const Expression& b) {return lit(a) >= b;};

    inline Expression operator&(const Expression& a, bool b) {return a & lit(b);};
    inline Expression operator&(bool a, const Expression& b) {return lit(a) & b;};

    inline Expression operator|(const Expression& a, bool b) {return a | lit(b);};
    inline Expression operator|(bool a, const Expression& b) {return lit(a) | b;};
}