#include "dataframelib/query_optimizer.h"
#include <unordered_set>
#include <algorithm>

namespace dataframelib {
    static void collect_columns(const ExprPtr& node, std::unordered_set<std::string>& seen, std::vector<std::string>& out) {
        if (!node) return;
        if (node->kind() == ExprKind::Column) {
            const auto& name = std::static_pointer_cast<ColumnExpr>(node)->name();
            if (seen.insert(name).second) {
                out.push_back(name);
            };
            return;
        };
        for (const auto& child : node->children()) {
            collect_columns(child, seen, out);
        };
    };
    static std::vector<std::string> extract_columns(const ExprPtr& node) {
        std::unordered_set<std::string> seen;
        std::vector<std::string> out;
        collect_columns(node, seen, out);
        return out;
    };
    auto schema_has_all = [](const std::shared_ptr<arrow::Schema>& schema, const std::vector<std::string>& cols) {
        for (const auto& col : cols) {
            if (schema->GetFieldIndex(col) == -1) return false;
        };
        return true;
    };
    static bool is_lit_numeric_value(const ExprPtr& expr, double target) {
        if (!expr || expr->kind() != ExprKind::Literal) return false;
        auto scalar = std::static_pointer_cast<LiteralExpr>(expr)->value();
        if (!scalar || !scalar->is_valid) return false;
        switch (scalar->type->id()) {
            case arrow::Type::INT8:   return std::static_pointer_cast<arrow::Int8Scalar>(scalar)->value   == static_cast<int8_t>(target);
            case arrow::Type::INT16:  return std::static_pointer_cast<arrow::Int16Scalar>(scalar)->value  == static_cast<int16_t>(target);
            case arrow::Type::INT32:  return std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value  == static_cast<int32_t>(target);
            case arrow::Type::INT64:  return std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value  == static_cast<int64_t>(target);
            case arrow::Type::UINT8:  return std::static_pointer_cast<arrow::UInt8Scalar>(scalar)->value  == static_cast<uint8_t>(target);
            case arrow::Type::UINT16: return std::static_pointer_cast<arrow::UInt16Scalar>(scalar)->value == static_cast<uint16_t>(target);
            case arrow::Type::UINT32: return std::static_pointer_cast<arrow::UInt32Scalar>(scalar)->value == static_cast<uint32_t>(target);
            case arrow::Type::UINT64: return std::static_pointer_cast<arrow::UInt64Scalar>(scalar)->value == static_cast<uint64_t>(target);
            case arrow::Type::FLOAT:  return std::static_pointer_cast<arrow::FloatScalar>(scalar)->value  == static_cast<float>(target);
            case arrow::Type::DOUBLE: return std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value == target;
            default: return false;
        };
    };
    static bool is_lit_bool_value(const ExprPtr& expr, bool target) {
        if (!expr || expr->kind() != ExprKind::Literal) return false;
        auto scalar = std::static_pointer_cast<LiteralExpr>(expr)->value();
        if (!scalar || !scalar->is_valid) return false;
        if (scalar->type->id() != arrow::Type::BOOL) return false;
        return std::static_pointer_cast<arrow::BooleanScalar>(scalar)->value == target;
    };
    std::shared_ptr<LogicalPlan> QueryOptimizer::optimize(std::shared_ptr<LogicalPlan> plan) {
        std::vector<std::shared_ptr<OptimizerRule>> rules = {
            std::make_shared<ConstFoldRule>(),
            std::make_shared<ExprSimplificationRule>(),
            std::make_shared<FilterEliminationRule>(),
            std::make_shared<ConjSplitRule>(),
            std::make_shared<PredicatePushdownRule>(),
            std::make_shared<ProjectionPushdownRule>(),
            std::make_shared<LimitPushdownRule>(),
            std::make_shared<ProjectMergingRule>(),
            std::make_shared<SortEliminationRule>(),
            std::make_shared<HavingVsWhereRule>()
        };
        auto current_plan = plan;
        int max_iterations = 10;
        for (int i = 0; i < max_iterations; ++i) {
            auto prev_plan = current_plan;
            for (const auto& rule : rules) {
                current_plan = recurse_children(current_plan, *rule);
            };
            if (current_plan == prev_plan) {
                break;
            };
        };
        std::vector<std::shared_ptr<OptimizerRule>> cleanup_rules = {
            std::make_shared<FilterMergingRule>(),
            std::make_shared<ProjectMergingRule>(),
            std::make_shared<SortEliminationRule>()
        };
        for (const auto& rule : cleanup_rules) {
            current_plan = recurse_children(current_plan, *rule);
        };
    return current_plan;
    };
    std::shared_ptr<LogicalPlan> QueryOptimizer::rebuild_with_children(
            const std::shared_ptr<LogicalPlan>& node,
            const std::vector<std::shared_ptr<LogicalPlan>>& new_children) {
        const auto& old = node->children();
        bool changed = old.size() != new_children.size();
        if (!changed) {
            for (size_t i = 0; i < old.size(); ++i) {
                if (old[i] != new_children[i]) {
                    changed = true;
                    break;
                };
            };
        };
        if (!changed) {
            return node;
        };
        switch (node->kind()) {
            case PlanKind::ScanCsv:
            case PlanKind::ScanParquet: {
                return node;
            };
            case PlanKind::Filter: {
                auto filter = std::static_pointer_cast<FilterPlan>(node);
                return std::make_shared<FilterPlan>(new_children[0], filter->predicate());
            };
            case PlanKind::Project: {
                auto project = std::static_pointer_cast<ProjectPlan>(node);
                if (project->is_name_only()) {
                    return std::make_shared<ProjectPlan>(new_children[0], project->name_cols());
                } 
                else {
                    return std::make_shared<ProjectPlan>(new_children[0], project->expr_cols());
                };
            };
            case PlanKind::Aggregate: {
                auto agg = std::dynamic_pointer_cast<AggregatePlan>(node);
                return std::make_shared<AggregatePlan>(new_children[0], agg->group_keys(), agg->aggr_exprs());
            };
            case PlanKind::Join: {
                auto join = std::dynamic_pointer_cast<JoinPlan>(node);
                return std::make_shared<JoinPlan>(new_children[0], new_children[1], join->on(), join->how());

            };
            case PlanKind::WithColumn: {
                auto wc = std::static_pointer_cast<WithColumnPlan>(node);
                return std::make_shared<WithColumnPlan>(new_children[0], wc->name(), wc->expr());
            };
            case PlanKind::Sort: {
                auto sort = std::static_pointer_cast<SortPlan>(node);
                return std::make_shared<SortPlan>(new_children[0], sort->columns(), sort->ascending());
            };
            case PlanKind::Limit: {
                auto limit = std::static_pointer_cast<LimitPlan>(node);
                return std::make_shared<LimitPlan>(new_children[0], limit->n());
            };
            case PlanKind::EmptyTable: {
                return node;
            };
            case PlanKind::TopK: {
                auto topk = std::static_pointer_cast<TopKPlan>(node);
                return std::make_shared<TopKPlan>(new_children[0], topk->columns(), topk->ascending(), topk->k());
            };
            default:
                throw std::runtime_error("Unknown plan kind in rebuild_with_children");
        };
    };
    std::shared_ptr<LogicalPlan> QueryOptimizer::recurse_children(
                                                std::shared_ptr<LogicalPlan> node, 
                                                const OptimizerRule& rule) {
        if (node->children().empty()) {
            return rule.apply(node);
        };
        std::vector<std::shared_ptr<LogicalPlan>> new_children;
        new_children.reserve(node->children().size());
        for (const auto& child : node->children()) {
            new_children.push_back(recurse_children(child, rule));
        };
        auto rebuilt = rebuild_with_children(node, new_children);
        return rule.apply(rebuilt);
    };
    std::shared_ptr<LogicalPlan> ConstFoldRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        switch (plan->kind()) {
            case PlanKind::Filter: {
                auto filter = std::static_pointer_cast<FilterPlan>(plan);
                auto folded = fold_expr(filter->predicate());
                if (folded.node() == filter->predicate().node()) return plan;
                return std::make_shared<FilterPlan>(filter->children()[0], folded);
            };
            case PlanKind::WithColumn: {
                auto wc = std::static_pointer_cast<WithColumnPlan>(plan);
                auto folded = fold_expr(wc->expr());
                if (folded.node() == wc->expr().node()) return plan;
                return std::make_shared<WithColumnPlan>(wc->children()[0], wc->name(), folded);
            };
            case PlanKind::Project: {
                auto proj = std::static_pointer_cast<ProjectPlan>(plan);
                if (proj->is_name_only()) return plan;
                std::vector<Expression> folded_exprs;
                folded_exprs.reserve(proj->expr_cols().size());
                bool changed = false;
                for (const auto& expr : proj->expr_cols()) {
                    auto folded = fold_expr(expr);
                    if (folded.node() != expr.node()) changed = true;
                    folded_exprs.push_back(folded);
                };
                if (!changed) return plan;
                return std::make_shared<ProjectPlan>(proj->children()[0], folded_exprs);
            };
            case PlanKind::Aggregate: {
                auto agg = std::static_pointer_cast<AggregatePlan>(plan);
                std::vector<Expression> folded_aggr_exprs;
                folded_aggr_exprs.reserve(agg->aggr_exprs().size());
                bool changed = false;
                for (const auto& expr : agg->aggr_exprs()) {
                    auto folded = fold_expr(expr);
                    if (folded.node() != expr.node()) changed = true;
                    folded_aggr_exprs.push_back(folded);
                };
                if (!changed) return plan;
                return std::make_shared<AggregatePlan>(agg->children()[0], agg->group_keys(), folded_aggr_exprs);
            };
            default:
                return plan;
        };
    };
    Expression ConstFoldRule::fold_expr(Expression expr) const {
        return Expression(fold_node(expr.node()));
    };
    ExprPtr ConstFoldRule::fold_node(ExprPtr node) const {
        if (!node) return nullptr;
        if (node->kind() == ExprKind::Literal || node->kind() == ExprKind::Column) {
            return node;
        };
        bool all_lit = true;
        bool children_changed = false;
        auto orig_children = node->children();
        std::vector<ExprPtr> folded_children;
        folded_children.reserve(orig_children.size());
        for (const auto& child : orig_children) {
            auto folded_child = fold_node(child);
            if (folded_child != child) children_changed = true;
            folded_children.push_back(folded_child);
            if (folded_child->kind() != ExprKind::Literal) {
                all_lit = false;
            };
        };
        if (all_lit && !folded_children.empty()) {
            auto dummy = arrow::Table::Make(arrow::schema({}), std::vector<std::shared_ptr<arrow::ChunkedArray>>{}, 1);
            switch (node->kind()) {
                case ExprKind::BinaryOp:
                case ExprKind::UnaryOp:
                case ExprKind::StringFn:
                case ExprKind::AggFn: {
                    auto eval_result = node->evaluate(dummy);
                    if (eval_result.ok()) {
                        auto datum = eval_result.ValueOrDie();
                        if (datum.is_scalar()) {
                            return std::make_shared<LiteralExpr>(datum.scalar());
                        };
                    };
                    break;
                };
                default:
                    break;
            };
        };
        if (!children_changed) return node;
        switch (node->kind()) {
            case ExprKind::Alias: {
                auto alias_node = std::static_pointer_cast<AliasExpr>(node);
                return std::make_shared<AliasExpr>(folded_children[0], alias_node->output_name());
            };
            case ExprKind::BinaryOp: {
                auto bin_node = std::static_pointer_cast<BinaryOpExpr>(node);
                return std::make_shared<BinaryOpExpr>(bin_node->op(), folded_children[0], folded_children[1]);
            };
            case ExprKind::UnaryOp: {
                auto unary_node = std::static_pointer_cast<UnaryOpExpr>(node);
                return std::make_shared<UnaryOpExpr>(unary_node->op(), folded_children[0]);
            };
            case ExprKind::StringFn: {
                auto str_node = std::static_pointer_cast<StringFnExpr>(node);
                return std::make_shared<StringFnExpr>(str_node->fn(), folded_children[0], str_node->arg());
            };
            case ExprKind::AggFn: {
                auto agg_node = std::static_pointer_cast<AggrExpr>(node);
                return std::make_shared<AggrExpr>(agg_node->fn(), folded_children[0]);
            };
            default:
                return node;
        };
    };
    std::shared_ptr<LogicalPlan> ExprSimplificationRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        switch (plan->kind()) {
            case PlanKind::Filter: {
                auto filter = std::static_pointer_cast<FilterPlan>(plan);
                auto simplified = simplify_expr(filter->predicate());
                if (simplified.node() == filter->predicate().node()) return plan;
                return std::make_shared<FilterPlan>(filter->children()[0], simplified);
            };
            case PlanKind::WithColumn: {
                auto wc = std::static_pointer_cast<WithColumnPlan>(plan);
                auto simplified = simplify_expr(wc->expr());
                if (simplified.node() == wc->expr().node()) return plan;
                return std::make_shared<WithColumnPlan>(wc->children()[0], wc->name(), simplified);
            };
            case PlanKind::Project: {
                auto proj = std::static_pointer_cast<ProjectPlan>(plan);
                if (proj->is_name_only()) return plan;
                std::vector<Expression> simplify_exprs;
                simplify_exprs.reserve(proj->expr_cols().size());
                bool changed = false;
                for (const auto& expr : proj->expr_cols()) {
                    auto simplified = simplify_expr(expr);
                    if (simplified.node() != expr.node()) changed = true;
                    simplify_exprs.push_back(simplified);
                };
                if (!changed) return plan;
                return std::make_shared<ProjectPlan>(proj->children()[0], simplify_exprs);
            };
            case PlanKind::Aggregate: {
                auto agg = std::static_pointer_cast<AggregatePlan>(plan);
                std::vector<Expression> simplify_aggr_exprs;
                simplify_aggr_exprs.reserve(agg->aggr_exprs().size());
                bool changed = false;
                for (const auto& expr : agg->aggr_exprs()) {
                    auto simplified = simplify_expr(expr);
                    if (simplified.node() != expr.node()) changed = true;
                    simplify_aggr_exprs.push_back(simplified);
                };
                if (!changed) return plan;
                return std::make_shared<AggregatePlan>(agg->children()[0], agg->group_keys(), simplify_aggr_exprs);
            };
            default:
                return plan;
        };
    };
    Expression ExprSimplificationRule::simplify_expr(Expression expr) const {
        return Expression(simplify_node(expr.node()));
    };
    ExprPtr ExprSimplificationRule::simplify_node(ExprPtr node) const {
        if (!node) return nullptr;
        auto orig_children = node->children();
        std::vector<ExprPtr> simplified_children;
        simplified_children.reserve(orig_children.size());
        bool children_changed = false;
        for (const auto& child : orig_children) {
            auto sim_child = simplify_node(child);
            if (sim_child != child) children_changed = true;
            simplified_children.push_back(sim_child);
        };
        switch (node->kind()) {
            case ExprKind::BinaryOp: {
                auto bin_node = std::static_pointer_cast<BinaryOpExpr>(node);
                auto op = bin_node->op();
                auto lhs = simplified_children[0];
                auto rhs = simplified_children[1];
                if (op == BinOp::Add) {
                    if (is_lit_numeric_value(lhs, 0)) return rhs;
                    if (is_lit_numeric_value(rhs, 0)) return lhs;
                }
                else if (op == BinOp::Sub) {
                    if (is_lit_numeric_value(rhs, 0)) return lhs;
                }
                else if (op == BinOp::Mul) {
                    if (is_lit_numeric_value(lhs, 1)) return rhs;
                    if (is_lit_numeric_value(rhs, 1)) return lhs;
                    if (is_lit_numeric_value(lhs, 0)) return lhs;
                    if (is_lit_numeric_value(rhs, 0)) return rhs;
                }
                else if (op == BinOp::Div) {
                    if (is_lit_numeric_value(rhs, 1)) return lhs;
                }
                else if (op == BinOp::And) {
                    if (is_lit_bool_value(lhs, true)) return rhs;
                    if (is_lit_bool_value(rhs, true)) return lhs;
                    if (is_lit_bool_value(lhs, false)) return lhs;
                    if (is_lit_bool_value(rhs, false)) return rhs;
                }
                else if (op == BinOp::Or) {
                    if (is_lit_bool_value(lhs, false)) return rhs;
                    if (is_lit_bool_value(rhs, false)) return lhs;
                    if (is_lit_bool_value(lhs, true)) return lhs;
                    if (is_lit_bool_value(rhs, true)) return rhs;
                };
                if (!children_changed) return node;
                return std::make_shared<BinaryOpExpr>(op, lhs, rhs);
            };
            case ExprKind::UnaryOp: {
                auto unary_node = std::static_pointer_cast<UnaryOpExpr>(node);
                auto op = unary_node->op();
                auto child = simplified_children[0];
                if (op == UnaryOp::Not && child->kind() == ExprKind::UnaryOp) {
                    auto inner = std::static_pointer_cast<UnaryOpExpr>(child);
                    if (inner->op() == UnaryOp::Not) {
                        return inner->children()[0];
                    };
                };
                if (op == UnaryOp::Negate && child->kind() == ExprKind::UnaryOp) {
                    auto inner = std::static_pointer_cast<UnaryOpExpr>(child);
                    if (inner->op() == UnaryOp::Negate) {
                        return inner->children()[0];
                    };
                };
                if (!children_changed) return node;
                return std::make_shared<UnaryOpExpr>(op, child);
            };
            case ExprKind::Alias: {
                if (!children_changed) return node;
                auto alias_node = std::static_pointer_cast<AliasExpr>(node);
                return std::make_shared<AliasExpr>(simplified_children[0], alias_node->output_name());
            };
            case ExprKind::AggFn: {
                if (!children_changed) return node;
                auto agg_node = std::static_pointer_cast<AggrExpr>(node);
                return std::make_shared<AggrExpr>(agg_node->fn(), simplified_children[0]);
            };
            default:
                return node;
        };
    };
    std::shared_ptr<LogicalPlan> FilterEliminationRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        if (plan->kind() != PlanKind::Filter) return plan;
        auto filter = std::static_pointer_cast<FilterPlan>(plan);
        auto pred_node = filter->predicate().node();
        if (pred_node->kind() == ExprKind::Literal) {
            auto lit_node = std::static_pointer_cast<LiteralExpr>(pred_node);
            auto scalar = lit_node->value();
            if (scalar->is_valid && scalar->type->id() == arrow::Type::BOOL) {
                auto bool_scalar = std::static_pointer_cast<arrow::BooleanScalar>(scalar);
                if (bool_scalar->value) {
                    return filter->children()[0];
                }
                else {
                    return std::make_shared<EmptyTablePlan>(filter->children()[0]->output_schema());
                };
            };
        };
        return plan;
    };
    std::shared_ptr<LogicalPlan> ConjSplitRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        if (plan->kind() != PlanKind::Filter) return plan;
        auto filter = std::static_pointer_cast<FilterPlan>(plan);
        auto pred_node = filter->predicate().node();
        if (pred_node->kind() == ExprKind::BinaryOp) {
            auto bin_node = std::static_pointer_cast<BinaryOpExpr>(pred_node);
            if (bin_node->op() == BinOp::And) {
                Expression left_expr(bin_node->children()[0]);
                Expression right_expr(bin_node->children()[1]);
                auto bottom_filter = std::make_shared<FilterPlan>(filter->children()[0], right_expr);
                return std::make_shared<FilterPlan>(bottom_filter, left_expr);
            };
        };
        return plan;
    };
    std::shared_ptr<LogicalPlan> PredicatePushdownRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        if (plan->kind() != PlanKind::Filter) return plan;
        auto filter = std::static_pointer_cast<FilterPlan>(plan);
        auto child = filter->children()[0];
        if (child->kind() == PlanKind::Sort) {
            auto sort_node = std::static_pointer_cast<SortPlan>(child);
            auto pushed_filter = std::make_shared<FilterPlan>(sort_node->children()[0], filter->predicate());
            return std::make_shared<SortPlan>(pushed_filter, sort_node->columns(), sort_node->ascending());
        };
        if (child->kind() == PlanKind::Project) {
            auto proj_node = std::static_pointer_cast<ProjectPlan>(child);
            if (proj_node->is_name_only()) {
                auto pushed_filter = std::make_shared<FilterPlan>(proj_node->children()[0], filter->predicate());
                return std::make_shared<ProjectPlan>(pushed_filter, proj_node->name_cols());
            };
        };
        if (child->kind() == PlanKind::Join) {
            auto join_node = std::static_pointer_cast<JoinPlan>(child);
            if (join_node->how() != "inner") return plan;
            std::shared_ptr<arrow::Schema> left_schema, right_schema;
            try {
                left_schema = join_node->children()[0]->output_schema();
                right_schema = join_node->children()[1]->output_schema();
            } catch (...) {
                return plan;
            }
            std::vector<std::string> used_cols = extract_columns(filter->predicate().node());
            if (schema_has_all(left_schema, used_cols)) {
                auto new_left = std::make_shared<FilterPlan>(join_node->children()[0], filter->predicate());
                return std::make_shared<JoinPlan>(new_left, join_node->children()[1], join_node->on(), join_node->how());
            }
            else if (schema_has_all(right_schema, used_cols)) {
                auto new_right = std::make_shared<FilterPlan>(join_node->children()[1], filter->predicate());
                return std::make_shared<JoinPlan>(join_node->children()[0], new_right, join_node->on(), join_node->how());
            };
        };
        return plan;
    };
    std::shared_ptr<LogicalPlan> ProjectionPushdownRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        if (plan->kind() != PlanKind::Project) return plan;
        auto proj = std::static_pointer_cast<ProjectPlan>(plan);
        if (!proj->is_name_only()) return plan;
        auto child = proj->children()[0];
        if (child->kind() == PlanKind::Filter) {
            auto filter_node = std::static_pointer_cast<FilterPlan>(child);
            std::vector<std::string> filter_cols = extract_columns(filter_node->predicate().node());
            bool safe_to_push = true;
            for (const auto& col : filter_cols) {
                if (std::find(proj->name_cols().begin(), proj->name_cols().end(), col) == proj->name_cols().end()) {
                    safe_to_push = false;
                    break;
                };
            };
            if (safe_to_push) {
                auto new_proj = std::make_shared<ProjectPlan>(filter_node->children()[0], proj->name_cols());
                return std::make_shared<FilterPlan>(new_proj, filter_node->predicate());
            };
        };
        if (child->kind() == PlanKind::Sort) {
            auto sorted_node = std::static_pointer_cast<SortPlan>(child);
            std::vector<std::string> sort_cols = sorted_node->columns();
            bool safe_to_push = true;
            for (const auto& col : sort_cols) {
                if (std::find(proj->name_cols().begin(), proj->name_cols().end(), col) == proj->name_cols().end()) {
                    safe_to_push = false;
                    break;
                };
            };
            if (safe_to_push) {
                auto new_proj = std::make_shared<ProjectPlan>(sorted_node->children()[0], proj->name_cols());
                return std::make_shared<SortPlan>(new_proj, sorted_node->columns(), sorted_node->ascending());
            };
        };
        return plan;
    };
    std::shared_ptr<LogicalPlan> LimitPushdownRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        if (plan->kind() != PlanKind::Limit) return plan;
        auto limit_node = std::static_pointer_cast<LimitPlan>(plan);
        auto child = limit_node->children()[0];
        if (child->kind() == PlanKind::Project) {
            auto proj_node = std::static_pointer_cast<ProjectPlan>(child);
            auto new_limit = std::make_shared<LimitPlan>(proj_node->children()[0], limit_node->n());
            if (proj_node->is_name_only()) {
                return std::make_shared<ProjectPlan>(new_limit, proj_node->name_cols());
            } 
            else {
                return std::make_shared<ProjectPlan>(new_limit, proj_node->expr_cols());
            };
        };
        if (child->kind() == PlanKind::Sort) {
            auto sort_node = std::static_pointer_cast<SortPlan>(child);
            return std::make_shared<TopKPlan>(
                sort_node->children()[0], 
                sort_node->columns(), 
                sort_node->ascending(), 
                limit_node->n()
            );
        };
        return plan;
    };
    std::shared_ptr<LogicalPlan> FilterMergingRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        if (plan->kind() != PlanKind::Filter) return plan;
        auto top_filter = std::static_pointer_cast<FilterPlan>(plan);
        if (top_filter->children()[0]->kind() == PlanKind::Filter) {
            auto bottom_filter = std::static_pointer_cast<FilterPlan>(top_filter->children()[0]);
            auto merged_pred = std::make_shared<BinaryOpExpr>(BinOp::And, top_filter->predicate().node(), bottom_filter->predicate().node());
            return std::make_shared<FilterPlan>(bottom_filter->children()[0], Expression(merged_pred));
        };
        return plan;
    };
    std::shared_ptr<LogicalPlan> ProjectMergingRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        if (plan->kind() != PlanKind::Project) return plan;
        auto top_proj = std::static_pointer_cast<ProjectPlan>(plan);
        if (top_proj->children()[0]->kind() == PlanKind::Project) {
            auto bottom_proj = std::static_pointer_cast<ProjectPlan>(top_proj->children()[0]);
            if (top_proj->is_name_only() && bottom_proj->is_name_only()) {
                return std::make_shared<ProjectPlan>(bottom_proj->children()[0], top_proj->name_cols());
            };
        };
        return plan;
    };
    std::shared_ptr<LogicalPlan> SortEliminationRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        if (plan->kind() != PlanKind::Sort) return plan;
        auto top_sort = std::static_pointer_cast<SortPlan>(plan);
        if (top_sort->children()[0]->kind() == PlanKind::Sort) {
            auto bottom_sort = std::static_pointer_cast<SortPlan>(top_sort->children()[0]);
            return std::make_shared<SortPlan>(bottom_sort->children()[0], top_sort->columns(), top_sort->ascending());
        };
        return plan;
    };
    std::shared_ptr<LogicalPlan> HavingVsWhereRule::apply(std::shared_ptr<LogicalPlan> plan) const {
        if (plan->kind() != PlanKind::Filter) return plan;
        auto filter = std::static_pointer_cast<FilterPlan>(plan);
        if (filter->children()[0]->kind() == PlanKind::Aggregate) {
            auto agg = std::static_pointer_cast<AggregatePlan>(filter->children()[0]);
            std::vector<std::string> filter_cols = extract_columns(filter->predicate().node());
            bool safe_to_push = true;
            for (const auto& col : filter_cols) {
                if (std::find(agg->group_keys().begin(), agg->group_keys().end(), col) == agg->group_keys().end()) {
                    safe_to_push = false;
                    break;
                };
            };
            if (safe_to_push) {
                auto pushed_filter = std::make_shared<FilterPlan>(agg->children()[0], filter->predicate());
                return std::make_shared<AggregatePlan>(pushed_filter, agg->group_keys(), agg->aggr_exprs());
            };
        };
        return plan;
    };
}