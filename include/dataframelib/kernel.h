#pragma once
#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <stdexcept>
#include <memory>
#include <cmath>
#include "expr.h"
#define DFL_CHECK_OK(expr) \
    do { \
        auto _s = (expr); \
        if (!_s.ok()) throw std::runtime_error(_s.ToString()); \
    } while(0)
namespace dataframelib {
    std::pair<std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Array>> 
    promote_arrays(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    template <typename V>
    auto dispatch_array(const std::shared_ptr<arrow::Array>& array, V&& visitor) {
        if (!array) {
            throw std::invalid_argument("Cannot dispatch a null array pointer");
        };
        switch (array->type_id()) {
            case arrow::Type::INT8:
                return visitor(std::static_pointer_cast<arrow::Int8Array>(array));
            case arrow::Type::INT16:
                return visitor(std::static_pointer_cast<arrow::Int16Array>(array));
            case arrow::Type::INT32:
                return visitor(std::static_pointer_cast<arrow::Int32Array>(array));
            case arrow::Type::INT64:
                return visitor(std::static_pointer_cast<arrow::Int64Array>(array));
            case arrow::Type::UINT8:
                return visitor(std::static_pointer_cast<arrow::UInt8Array>(array));
            case arrow::Type::UINT16:
                return visitor(std::static_pointer_cast<arrow::UInt16Array>(array));
            case arrow::Type::UINT32:
                return visitor(std::static_pointer_cast<arrow::UInt32Array>(array));
            case arrow::Type::UINT64:
                return visitor(std::static_pointer_cast<arrow::UInt64Array>(array));
            case arrow::Type::FLOAT:
                return visitor(std::static_pointer_cast<arrow::FloatArray>(array));
            case arrow::Type::DOUBLE:
                return visitor(std::static_pointer_cast<arrow::DoubleArray>(array));
            case arrow::Type::BOOL:
                return visitor(std::static_pointer_cast<arrow::BooleanArray>(array));
            case arrow::Type::STRING:
                return visitor(std::static_pointer_cast<arrow::StringArray>(array));
            default:
                throw std::runtime_error(
                    std::string("Unsupported Arrow type for kernel dispatch: ") +
                    (array->type() ? array->type()->ToString() : std::string("<null type>"))
                );
        };
    };
    struct AddOp {
        template <typename T>
        T operator()(T a, T b) const { 
            return a + b; 
        };
    };
    struct SubtractOp {
        template <typename T>
        T operator()(T a, T b) const { 
            return a - b; 
        };
    };
    struct MultiplyOp {
        template <typename T>
        T operator()(T a, T b) const { 
            return a * b; 
        };
    };
    struct DivideOp {
        template <typename T>
        T operator()(T a, T b) const {
            return a / b;
        };
    };
    struct ModuloOp {
        template <typename T>
        T operator()(T a, T b) const {
            if constexpr (std::is_floating_point_v<T>) {
                return std::fmod(a, b);
            } 
            else {
                return a % b;
            };
        };
    };
    struct EqOp {
        template <typename T>
        bool operator()(T a, T b) const { 
            return a == b; 
        };
    };
    struct NeqOp {
        template <typename T>
        bool operator()(T a, T b) const { 
            return a != b; 
        };
    };
    struct GtOp {
        template <typename T>
        bool operator()(T a, T b) const { 
            return a > b; 
        };
    };
    struct LtOp {
        template <typename T>
        bool operator()(T a, T b) const { 
            return a < b;
        };
    };
    struct GteOp {
        template <typename T>
        bool operator()(T a, T b) const { 
            return a >= b; 
        };
    };
    struct LteOp {
        template <typename T>
        bool operator()(T a, T b) const { 
            return a <= b; 
        };
    };
    struct AndKleeneOp {
        static void apply(uint8_t v_a, uint8_t v_b, uint8_t d_a, uint8_t d_b, uint8_t& v_out, uint8_t& d_out) {
            d_out = d_a & d_b;
            v_out = (v_a & v_b) | (v_a & ~d_a) | (v_b & ~d_b);
        };
    };
    struct OrKleeneOp {
        static void apply(uint8_t v_a, uint8_t v_b, uint8_t d_a, uint8_t d_b, uint8_t& v_out, uint8_t& d_out) {
            d_out = d_a | d_b;
            v_out = (v_a & v_b) | (v_a & d_a) | (v_b & d_b);
        };
    };
    struct NotKleeneOp {
        static void apply(uint8_t v_a, uint8_t d_a, uint8_t& v_out, uint8_t& d_out) {
            d_out = ~d_a;
            v_out = v_a;
        };
    };
    struct AbsOp {
        template <typename T>
        T operator()(T a) const { 
            return a < 0 ? -a : a; 
        };
    };
    struct NegateOp {
        template <typename T>
        T operator()(T a) const { 
            return -a; 
        };
    };
    struct Utf8LengthOp {
        int64_t operator()(std::string_view a) const { 
            return a.length(); 
        };
    };
    struct ContainsOp {
        bool operator()(std::string_view a, std::string_view b) const { 
            return a.find(b) != std::string_view::npos; 
        };
    };
    struct StartsWithOp {
        bool operator()(std::string_view a, std::string_view b) const { 
            return a.length() >= b.length() && a.substr(0, b.length()) == b; 
        };
    };
    struct EndsWithOp {
        bool operator()(std::string_view a, std::string_view b) const { 
            return a.length() >= b.length() && a.substr(a.length() - b.length()) == b; 
        };
    };
    struct ToLowerOp {
        mutable std::string buffer; 
        std::string_view operator()(std::string_view input) const {
            buffer.resize(input.size()); 
            std::transform(input.begin(), input.end(), buffer.begin(), 
                [](unsigned char c){ return std::tolower(c); });
            return std::string_view(buffer); 
        };
    };
    struct ToUpperOp {
        mutable std::string buffer; 
        std::string_view operator()(std::string_view input) const {
            buffer.resize(input.size()); 
            std::transform(input.begin(), input.end(), buffer.begin(), 
                [](unsigned char c){ return std::toupper(c); });
            return std::string_view(buffer);
        };
    };
    struct SumAggOp {
        template <typename T> 
        static T accumulate(T current, T next_val) { 
            return current + next_val; 
        };
        template <typename T>
        static std::shared_ptr<arrow::Scalar> finalize(T sum, int64_t valid_count) {
            if (valid_count == 0) {
                return arrow::MakeNullScalar(arrow::TypeTraits<typename arrow::CTypeTraits<T>
                                                            ::ArrowType>::type_singleton());
            };
            return std::make_shared<typename arrow::CTypeTraits<T>::ScalarType>(sum);
        };
    };
    struct MeanAggOp {
        template <typename T>
        static T accumulate(T current, T next_val) {
            return current + next_val;
        };
        template <typename T>
        static std::shared_ptr<arrow::Scalar> finalize(T sum, int64_t valid_count) {
            if (valid_count == 0) {
                return arrow::MakeNullScalar(arrow::TypeTraits<typename arrow::CTypeTraits<T>::ArrowType>::type_singleton());
            };
            return std::make_shared<typename arrow::CTypeTraits<T>::ScalarType>(sum / valid_count);
        };
    };
    struct CountAggOp {
        template <typename T> 
        static T accumulate(T current, T next_val) { 
            return current+1; 
        };
        template <typename T>
        static std::shared_ptr<arrow::Scalar> finalize(T sum, int64_t valid_count) {
            return std::make_shared<arrow::Int64Scalar>(valid_count);
        };
    };
    struct MinAggOp {
        template <typename T> 
        static T accumulate(T current, T next_val) { 
            return std::min(current, next_val); 
        };
        template <typename T>
        static std::shared_ptr<arrow::Scalar> finalize(T min, int64_t valid_count) {
            if (valid_count == 0) {
                return arrow::MakeNullScalar(arrow::TypeTraits<typename arrow::CTypeTraits<T>
                                                            ::ArrowType>::type_singleton());
            };
            return std::make_shared<typename arrow::CTypeTraits<T>::ScalarType>(min);
        };
    };
    struct MaxAggOp {
        template <typename T> 
        static T accumulate(T current, T next_val) { 
            return std::max(current, next_val); 
        };
        template <typename T>
        static std::shared_ptr<arrow::Scalar> finalize(T max, int64_t valid_count) {
            if (valid_count == 0) {
                return arrow::MakeNullScalar(arrow::TypeTraits<typename arrow::CTypeTraits<T>
                                                            ::ArrowType>::type_singleton());
            };
            return std::make_shared<typename arrow::CTypeTraits<T>::ScalarType>(max);
        };
    };
    template <typename Op, typename BuilderType>
    struct BinaryNumericKernel {
        static std::shared_ptr<arrow::Array> apply(
            const std::shared_ptr<arrow::Array>& lhs,
            const std::shared_ptr<arrow::Array>& rhs) {
            BuilderType builder;
            Op op;
            using ArrayType = typename arrow::TypeTraits<typename BuilderType::TypeClass>::ArrayType;
            auto left = std::static_pointer_cast<ArrayType>(lhs);
            auto right = std::static_pointer_cast<ArrayType>(rhs);
            const int64_t n = left->length();
            DFL_CHECK_OK(builder.Reserve(n));
            for (int64_t i = 0; i < n; ++i) {
                if (left->IsNull(i) || right->IsNull(i)) {
                    builder.UnsafeAppendNull();
                    continue;
                };
                auto l_val = left->Value(i);
                auto r_val = right->Value(i);
                if constexpr (std::is_same_v<Op, DivideOp> || std::is_same_v<Op, ModuloOp>) {
                    if (r_val == 0) {
                        builder.UnsafeAppendNull();
                        continue;
                    };
                };
                builder.UnsafeAppend(op(l_val, r_val));
            };
            std::shared_ptr<arrow::Array> out;
            DFL_CHECK_OK(builder.Finish(&out));
            return out;
        };
    };
    template <typename Op, typename InputArrayType>
    struct BinaryComparisonKernel {
        static std::shared_ptr<arrow::Array> apply(
            const std::shared_ptr<arrow::Array>& lhs,
            const std::shared_ptr<arrow::Array>& rhs) {
            arrow::BooleanBuilder builder;
            Op op;
            auto left = std::static_pointer_cast<InputArrayType>(lhs);
            auto right = std::static_pointer_cast<InputArrayType>(rhs);
            const int64_t n = left->length();
            DFL_CHECK_OK(builder.Reserve(n));
            for (int64_t i = 0; i < n; ++i) {
                if (left->IsNull(i) || right->IsNull(i)) {
                    builder.UnsafeAppendNull();
                    continue;
                };
                builder.UnsafeAppend(op(left->Value(i), right->Value(i)));
            };
            std::shared_ptr<arrow::Array> out;
            DFL_CHECK_OK(builder.Finish(&out));
            return out;
        };
    };
    template <typename KleeneOp>
    struct BinaryKleeneKernel {
        static std::shared_ptr<arrow::Array> apply(
            const std::shared_ptr<arrow::Array>& lhs, 
            const std::shared_ptr<arrow::Array>& rhs) {
            auto left = (lhs->offset() == 0) ? std::static_pointer_cast<arrow::BooleanArray>(lhs) 
                : std::static_pointer_cast<arrow::BooleanArray>(arrow::Concatenate({lhs}).ValueOrDie());
            auto right = (rhs->offset() == 0) ? std::static_pointer_cast<arrow::BooleanArray>(rhs) 
                : std::static_pointer_cast<arrow::BooleanArray>(arrow::Concatenate({rhs}).ValueOrDie());
            int64_t length = left->length();
            std::shared_ptr<arrow::Buffer> out_data_buf;
            std::shared_ptr<arrow::Buffer> out_valid_buf;
            DFL_CHECK_OK(arrow::AllocateBitmap(length).Value(&out_data_buf));
            DFL_CHECK_OK(arrow::AllocateBitmap(length).Value(&out_valid_buf));
            const uint8_t* left_data = left->values()->data();
            const uint8_t* right_data = right->values()->data();
            const uint8_t* left_valid = left->null_bitmap_data();
            const uint8_t* right_valid = right->null_bitmap_data();
            uint8_t* out_data = out_data_buf->mutable_data();
            uint8_t* out_valid = out_valid_buf->mutable_data();
            int64_t num_bytes = arrow::bit_util::BytesForBits(length);
            for (int64_t i = 0; i < num_bytes; ++i) {
                uint8_t v_a = left_valid ? left_valid[i] : 0xFF;
                uint8_t v_b = right_valid ? right_valid[i] : 0xFF;
                uint8_t d_a = left_data[i];
                uint8_t d_b = right_data[i];
                KleeneOp::apply(v_a, v_b, d_a, d_b, out_valid[i], out_data[i]);
            };
            return std::make_shared<arrow::BooleanArray>(length, out_data_buf, out_valid_buf);
        };
    };
    template <typename Op, typename BuilderType>
    struct UnaryNumericKernel {
        static std::shared_ptr<arrow::Array> apply(const std::shared_ptr<arrow::Array>& input) {
            BuilderType builder;
            Op op;
            using ArrayType = typename arrow::TypeTraits<typename BuilderType::TypeClass>::ArrayType;
            auto typed_array = std::static_pointer_cast<ArrayType>(input);
            const int64_t n = typed_array->length();
            DFL_CHECK_OK(builder.Reserve(n));
            for (int64_t i = 0; i < n; ++i) {
                if (typed_array->IsNull(i)) builder.UnsafeAppendNull();
                else builder.UnsafeAppend(op(typed_array->Value(i)));
            };
            std::shared_ptr<arrow::Array> out;
            DFL_CHECK_OK(builder.Finish(&out));
            return out;
        };
    };
    template <typename UnaryOp, typename InputArrayType>
    struct UnaryKleeneKernel {
        static std::shared_ptr<arrow::Array> apply(const std::shared_ptr<arrow::Array>& input) {
            auto arr = (input->offset() == 0) ? std::static_pointer_cast<InputArrayType>(input) 
                : std::static_pointer_cast<InputArrayType>(arrow::Concatenate({input}).ValueOrDie());
            int64_t length = arr->length();      
            std::shared_ptr<arrow::Buffer> out_data_buf;
            std::shared_ptr<arrow::Buffer> out_valid_buf;
            DFL_CHECK_OK(arrow::AllocateBitmap(length).Value(&out_data_buf));
            DFL_CHECK_OK(arrow::AllocateBitmap(length).Value(&out_valid_buf));
            const uint8_t* data = arr->values()->data();
            const uint8_t* valid = arr->null_bitmap_data();
            uint8_t* out_data = out_data_buf->mutable_data();
            uint8_t* out_valid = out_valid_buf->mutable_data();
            int64_t num_bytes = arrow::bit_util::BytesForBits(length);
            for (int64_t i = 0; i < num_bytes; ++i) {
                uint8_t v_a = valid ? valid[i] : 0xFF;
                uint8_t d_a = data[i];
                UnaryOp::apply(v_a, d_a, out_valid[i], out_data[i]);
            };
            return std::make_shared<arrow::BooleanArray>(length, out_data_buf, out_valid_buf);
        };
    };
    template <typename Op, typename BuilderType>
    struct UnaryStringKernel {
        static std::shared_ptr<arrow::Array> apply(const std::shared_ptr<arrow::Array>& input) {
            auto string_array = std::static_pointer_cast<arrow::StringArray>(input);
            BuilderType builder;
            Op op;
            const int64_t n = string_array->length();
            DFL_CHECK_OK(builder.Reserve(n));
            for (int64_t i = 0; i < n; ++i) {
                if (string_array->IsNull(i)) DFL_CHECK_OK(builder.AppendNull());
                else DFL_CHECK_OK(builder.Append(op(string_array->GetView(i))));
            };
            std::shared_ptr<arrow::Array> out;
            DFL_CHECK_OK(builder.Finish(&out));
            return out;
        };
    };
    template <typename Op>
    struct StringPredicateKernel {
        static std::shared_ptr<arrow::Array> apply(const std::shared_ptr<arrow::Array>& input, std::string_view arg) {
            auto string_array = std::static_pointer_cast<arrow::StringArray>(input);
            arrow::BooleanBuilder builder;
            Op op;
            const int64_t n = string_array->length();
            DFL_CHECK_OK(builder.Reserve(n));
            for (int64_t i = 0; i < n; ++i) {
                if (string_array->IsNull(i)) builder.UnsafeAppendNull();
                else builder.UnsafeAppend(op(string_array->GetView(i), arg));
            };
            std::shared_ptr<arrow::Array> out;
            DFL_CHECK_OK(builder.Finish(&out));
            return out;
        };
    };
    template <typename Op>
    struct BinaryStringComparisonKernel {
        static std::shared_ptr<arrow::Array> apply(
            const std::shared_ptr<arrow::Array>& lhs,
            const std::shared_ptr<arrow::Array>& rhs) {
            arrow::BooleanBuilder builder;
            Op op;
            auto left  = std::static_pointer_cast<arrow::StringArray>(lhs);
            auto right = std::static_pointer_cast<arrow::StringArray>(rhs);
            const int64_t n = left->length();
            DFL_CHECK_OK(builder.Reserve(n));
            for (int64_t i = 0; i < n; ++i) {
                if (left->IsNull(i) || right->IsNull(i)) {
                    builder.UnsafeAppendNull();
                    continue;
                };
                builder.UnsafeAppend(op(left->GetView(i), right->GetView(i)));
            };
            std::shared_ptr<arrow::Array> out;
            DFL_CHECK_OK(builder.Finish(&out));
            return out;
        };
    };
    template <typename Op, typename ArrayType>
    struct AggregateNumericKernel {
        static std::shared_ptr<arrow::Scalar> apply(
            const std::shared_ptr<arrow::Array>& input,
            const std::vector<int64_t>& indices) {
            auto typed_array = std::static_pointer_cast<ArrayType>(input);
            int64_t valid_count = 0;
            if constexpr (std::is_same_v<Op, CountAggOp>) {
                for (int64_t idx : indices) {
                    if (typed_array->IsNull(idx)) continue;
                    valid_count++;
                };
                return Op::template finalize<int64_t>(0, valid_count);
            }
            else {
                if constexpr (arrow::is_number_type<typename ArrayType::TypeClass>::value) {
                    using ValType = typename arrow::TypeTraits<typename ArrayType::TypeClass>::CType;
                    ValType state = 0;
                    for (int64_t idx : indices) {
                        if (typed_array->IsNull(idx)) continue;
                        if (valid_count == 0) {
                            state = typed_array->Value(idx);
                        }
                        else {
                            state = Op::accumulate(state, typed_array->Value(idx));
                        };
                        valid_count++;
                    };
                    return Op::finalize(state, valid_count);
                } 
                else if constexpr (arrow::is_boolean_type<typename ArrayType::TypeClass>::value) {
                    using ValType = bool;
                    ValType state = 0;
                    for (int64_t idx : indices) {
                        if (typed_array->IsNull(idx)) continue;
                        if (valid_count == 0) {
                            state = typed_array->Value(idx);
                        }
                        else {
                            state = Op::accumulate(state, typed_array->Value(idx));
                        };
                        valid_count++;
                    };
                    return Op::finalize(state, valid_count);
                } 
                else {
                    throw std::runtime_error("Unsupported aggregation type");
                };
            };
        };
    };
    std::shared_ptr<arrow::Array> add(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> sub(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> mul(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> div(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> mod(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> eq(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> ne(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> lt(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> le(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> gt(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> ge(const std::shared_ptr<arrow::Array>& left, const std::shared_ptr<arrow::Array>& right);
    std::shared_ptr<arrow::Array> and_kleene(const std::shared_ptr<arrow::Array>& lhs, const std::shared_ptr<arrow::Array>& rhs);
    std::shared_ptr<arrow::Array> or_kleene(const std::shared_ptr<arrow::Array>& lhs, const std::shared_ptr<arrow::Array>& rhs);
    std::shared_ptr<arrow::Array> not_kleene(const std::shared_ptr<arrow::Array>& input);
    std::shared_ptr<arrow::Array> abs(const std::shared_ptr<arrow::Array>& input);
    std::shared_ptr<arrow::Array> negate(const std::shared_ptr<arrow::Array>& input);
    std::shared_ptr<arrow::Array> is_null(const std::shared_ptr<arrow::Array>& input);
    std::shared_ptr<arrow::Array> is_not_null(const std::shared_ptr<arrow::Array>& input);
    std::shared_ptr<arrow::Array> utf8_length(const std::shared_ptr<arrow::Array>& input);
    std::shared_ptr<arrow::Array> contains(const std::shared_ptr<arrow::Array>& input, std::string_view substring);
    std::shared_ptr<arrow::Array> starts_with(const std::shared_ptr<arrow::Array>& input, std::string_view prefix);
    std::shared_ptr<arrow::Array> ends_with(const std::shared_ptr<arrow::Array>& input, std::string_view suffix);
    std::shared_ptr<arrow::Array> to_lower(const std::shared_ptr<arrow::Array>& input);
    std::shared_ptr<arrow::Array> to_upper(const std::shared_ptr<arrow::Array>& input);
    std::shared_ptr<arrow::Array> broadcast_scalar(const std::shared_ptr<arrow::Scalar>& scalar, int64_t length);
    std::shared_ptr<arrow::Scalar> AggregateKernel(
        AggFn fn, 
        const std::shared_ptr<arrow::Array>& values, 
        const std::vector<int64_t>& indices
    );
}