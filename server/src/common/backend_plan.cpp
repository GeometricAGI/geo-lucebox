#include "backend_plan_internal.h"

#include "feature_gate.h"
#include "model_capabilities.h"

#include <limits>
#include <utility>

namespace dflash::common::detail {

namespace {

BackendPreparationFailure failure(
    BackendPreparationError error,
    std::string message,
    std::vector<std::string> warnings = {}) {
    return {error, std::move(message), std::move(warnings)};
}

PlacementBackend resolve_target_backend(
    const BackendArgs & args,
    PlacementBackend compiled_backend) {
    return args.device.backend == PlacementBackend::Auto
        ? compiled_backend
        : args.device.backend;
}

}  // namespace

BackendPreparation BackendPlanBuilder::resolve(
    BackendArgs args,
    BackendAdmissionContext admission,
    GgufModelInfo model,
    PlacementBackend compiled_backend) {
    if (args.model_path.empty()) {
        return failure(
            BackendPreparationError::InvalidRequest,
            "model_path is empty");
    }

    if (model.arch.empty()) {
        return failure(
            BackendPreparationError::ModelInspection,
            "failed to detect architecture from " + args.model_path);
    }

    if (!arch_is_supported(model.arch)) {
        return failure(
            BackendPreparationError::ModelInspection,
            "unsupported model architecture '" + model.arch + "' in " +
                args.model_path);
    }

    const PlacementBackend target_backend =
        resolve_target_backend(args, compiled_backend);

    if (args.specla_mode && !args.ddtree_tau_explicit) {
        args.ddtree_tau = 6.0f;
    }

    // Preserve the original admission order. Warnings describe what the
    // operator requested, before model-specific SpecLA normalization changes
    // the effective construction request.
    std::string incompatible = check_feature_compatibility(
        args, admission, model.arch, target_backend, compiled_backend);
    if (!incompatible.empty()) {
        return failure(
            BackendPreparationError::FeatureCompatibility,
            std::move(incompatible));
    }

    std::vector<std::string> warnings =
        collect_feature_warnings(args, admission, model.arch);

    BackendPlan::SpeclaEnvironmentAction specla_environment =
        BackendPlan::SpeclaEnvironmentAction::Preserve;

    if (args.specla_mode) {
        const bool supported =
            model.arch == "qwen35" && !args.device.is_multi_device();
        if (supported) {
            if (!args.draft_path.has_value()) {
                return failure(
                    BackendPreparationError::FeatureCompatibility,
                    "Qwen3.6 SpecLA requires --draft <path>",
                    std::move(warnings));
            }

            args.ddtree_mode = true;
            if (admission.kvflash_requested()) {
                warnings.push_back(
                    "--specla is unavailable with KVFlash; using ordinary "
                    "DDTree verification");
                args.specla_mode = false;
                if (!args.ddtree_tau_explicit) {
                    args.ddtree_tau = std::numeric_limits<float>::infinity();
                }
                specla_environment =
                    BackendPlan::SpeclaEnvironmentAction::Disable;
            } else {
                specla_environment =
                    BackendPlan::SpeclaEnvironmentAction::Enable;
            }
        } else {
            warnings.push_back(
                "--specla is unavailable for architecture '" + model.arch +
                "' with placement " + placement_device_name(args.device) +
                "; using the architecture's normal decode path");
            args.specla_mode = false;
            if (!args.ddtree_tau_explicit) {
                args.ddtree_tau = std::numeric_limits<float>::infinity();
            }
            specla_environment =
                BackendPlan::SpeclaEnvironmentAction::Disable;
        }
    }

    // Validate the exact snapshot construction will consume. This replaces
    // the old factory recheck against a second mutable BackendArgs object.
    incompatible = check_feature_compatibility(
        args, admission, model.arch, target_backend, compiled_backend);
    if (!incompatible.empty()) {
        return failure(
            BackendPreparationError::FeatureCompatibility,
            std::move(incompatible),
            std::move(warnings));
    }

    BackendPlan plan;
    plan.args_ = std::move(args);
    plan.model_ = std::move(model);
    plan.warnings_ = std::move(warnings);
    plan.specla_environment_ = specla_environment;
    return plan;
}

}  // namespace dflash::common::detail
