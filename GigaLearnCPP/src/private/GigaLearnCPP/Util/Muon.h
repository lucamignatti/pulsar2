#pragma once
#include "../FrameworkTorch.h"

#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

namespace GGL {

	struct MuonOptions : torch::optim::OptimizerOptions {
		typedef std::tuple<double, double, double> NSCoefficients;
		typedef std::tuple<double, double> AdamWBetas;

		explicit MuonOptions(double lr = 1e-3) : lr_(lr) {}

		TORCH_ARG(double, lr) = 1e-3;
		TORCH_ARG(double, weight_decay) = 0.1;
		TORCH_ARG(double, momentum) = 0.95;
		TORCH_ARG(bool, nesterov) = true;
		TORCH_ARG(double, eps) = 1e-7;
		TORCH_ARG(double, adamw_eps) = 1e-8;
		TORCH_ARG(AdamWBetas, adamw_betas) = std::make_tuple(0.9, 0.999);
		TORCH_ARG(int64_t, ns_steps) = 5;
		TORCH_ARG(NSCoefficients, ns_coefficients) = std::make_tuple(3.4445, -4.775, 2.0315);

	public:
		std::unique_ptr<torch::optim::OptimizerOptions> clone() const override {
			return std::make_unique<MuonOptions>(*this);
		}

		double get_lr() const override {
			return lr();
		}

		void set_lr(const double newLR) override {
			lr(newLR);
		}

		void serialize(torch::serialize::InputArchive& archive) override {
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, momentum);
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, nesterov);
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, adamw_eps);
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, ns_steps);

			c10::IValue ivalue;
			if (archive.try_read("adamw_beta1", ivalue)) {
				double beta1 = ivalue.toDouble();
				archive.read("adamw_beta2", ivalue);
				adamw_betas(std::make_tuple(beta1, ivalue.toDouble()));
			}
			if (archive.try_read("ns_a", ivalue)) {
				double a = ivalue.toDouble();
				archive.read("ns_b", ivalue);
				double b = ivalue.toDouble();
				archive.read("ns_c", ivalue);
				ns_coefficients(std::make_tuple(a, b, ivalue.toDouble()));
			}
		}

		void serialize(torch::serialize::OutputArchive& archive) const override {
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum);
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(nesterov);
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(adamw_eps);
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(ns_steps);

			auto [beta1, beta2] = adamw_betas();
			archive.write("adamw_beta1", torch::IValue(beta1));
			archive.write("adamw_beta2", torch::IValue(beta2));

			auto [a, b, c] = ns_coefficients();
			archive.write("ns_a", torch::IValue(a));
			archive.write("ns_b", torch::IValue(b));
			archive.write("ns_c", torch::IValue(c));
		}
	};

	struct MuonParamState : torch::optim::OptimizerParamState {
		TORCH_ARG(torch::Tensor, momentum_buffer);
		TORCH_ARG(int64_t, step) = 0;
		TORCH_ARG(torch::Tensor, exp_avg);
		TORCH_ARG(torch::Tensor, exp_avg_sq);

	public:
		std::unique_ptr<torch::optim::OptimizerParamState> clone() const override {
			return std::make_unique<MuonParamState>(*this);
		}

		void serialize(torch::serialize::InputArchive& archive) override {
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, momentum_buffer);
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg);
			_TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg_sq);
		}

		void serialize(torch::serialize::OutputArchive& archive) const override {
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum_buffer);
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
			_TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
		}
	};

	class Muon : public torch::optim::Optimizer {
	public:
		explicit Muon(const std::vector<torch::optim::OptimizerParamGroup>& paramGroups, MuonOptions defaults)
			: Optimizer(paramGroups, std::unique_ptr<torch::optim::OptimizerOptions>(new MuonOptions(defaults))) {
			ValidateOptions(defaults);
		}

		explicit Muon(std::vector<torch::Tensor> params, MuonOptions defaults)
			: Muon({ torch::optim::OptimizerParamGroup(std::move(params)) }, std::move(defaults)) {}

		torch::Tensor step(LossClosure closure = nullptr) override {
			torch::Tensor loss = {};
			if (closure != nullptr) {
				at::AutoGradMode enableGrad(true);
				loss = closure();
			}

			RG_NO_GRAD;
			for (auto& group : param_groups()) {
				auto& options = static_cast<MuonOptions&>(group.options());
				ValidateOptions(options);

				for (auto& param : group.params()) {
					if (!param.grad().defined())
						continue;

					auto grad = param.grad().detach();

					auto* paramID = param.unsafeGetTensorImpl();
					auto& stateEntry = state()[paramID];
					if (stateEntry == nullptr)
						stateEntry = std::make_unique<MuonParamState>();

					auto& paramState = static_cast<MuonParamState&>(*stateEntry);
					if (grad.dim() < 2) {
						StepAdamW(param, grad, options, paramState);
						continue;
					}

					if (!paramState.momentum_buffer().defined())
						paramState.momentum_buffer(torch::zeros_like(grad));

					auto buffer = paramState.momentum_buffer();
					buffer.mul_(options.momentum()).add_(grad);

					auto update = options.nesterov()
						? grad.add(buffer, options.momentum())
						: buffer;

					auto matrix = update.view({ update.size(0), -1 });
					auto orthoUpdate = ZeroPowerViaNewtonSchulz(matrix, options).view_as(update);

					if (options.weight_decay() != 0.0)
						param.mul_(1.0 - options.lr() * options.weight_decay());

					param.add_(orthoUpdate, -AdjustedLR(options.lr(), update));
				}
			}

			return loss;
		}

		void save(torch::serialize::OutputArchive& archive) const override {
			torch::optim::serialize<MuonParamState, MuonOptions>(archive, *this);
		}

		void load(torch::serialize::InputArchive& archive) override {
			torch::optim::serialize<MuonParamState, MuonOptions>(archive, *this);
		}

	private:
		static void ValidateOptions(const MuonOptions& options) {
			TORCH_CHECK(options.lr() >= 0, "Invalid learning rate: ", options.lr());
			TORCH_CHECK(options.weight_decay() >= 0, "Invalid weight_decay value: ", options.weight_decay());
			TORCH_CHECK(options.momentum() >= 0, "Invalid momentum value: ", options.momentum());
			TORCH_CHECK(options.eps() >= 0, "Invalid epsilon value: ", options.eps());
			TORCH_CHECK(options.adamw_eps() >= 0, "Invalid AdamW epsilon value: ", options.adamw_eps());
			TORCH_CHECK(options.ns_steps() >= 0, "Invalid ns_steps value: ", options.ns_steps());
		}

		static double AdjustedLR(double lr, const torch::Tensor& tensor) {
			double rows = (double)tensor.size(0);
			double cols = (double)(tensor.numel() / tensor.size(0));
			// RMS-match the orthogonalized update to an Adam-scale step so the
			// existing Adam-tuned LR (e.g. 1.5e-4) is effective. The NS5 output is
			// semi-orthogonal (per-element magnitude ~1/sqrt(d)), so canonical
			// Muon's sqrt(max(1, rows/cols)) scaling (==1 for square layers) makes
			// each step ~sqrt(d) (~17x for 256-wide) smaller than Adam at the same
			// LR -- which froze the policy (KL ~1e-7, entropy pinned at max).
			// sqrt(max(rows, cols)) restores ~Adam-RMS per-element magnitude.
			return lr * std::sqrt(std::max(rows, cols));
		}

		static void StepAdamW(torch::Tensor& param, const torch::Tensor& grad, const MuonOptions& options, MuonParamState& state) {
			if (!state.exp_avg().defined())
				state.exp_avg(torch::zeros_like(grad));
			if (!state.exp_avg_sq().defined())
				state.exp_avg_sq(torch::zeros_like(grad));

			state.step(state.step() + 1);

			auto [beta1, beta2] = options.adamw_betas();
			if (options.weight_decay() != 0.0)
				param.mul_(1.0 - options.lr() * options.weight_decay());

			auto expAvg = state.exp_avg();
			auto expAvgSq = state.exp_avg_sq();
			expAvg.mul_(beta1).add_(grad, 1.0 - beta1);
			expAvgSq.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

			double biasCorrection1 = 1.0 - std::pow(beta1, state.step());
			double biasCorrection2 = 1.0 - std::pow(beta2, state.step());
			double stepSize = options.lr() / biasCorrection1;
			auto denom = expAvgSq.sqrt() / std::sqrt(biasCorrection2) + options.adamw_eps();
			param.addcdiv_(expAvg, denom, -stepSize);
		}

		static torch::Tensor ZeroPowerViaNewtonSchulz(torch::Tensor grad, const MuonOptions& options) {
			auto X = grad.to(torch::kFloat32);
			const bool transposed = X.size(0) > X.size(1);
			if (transposed)
				X = X.transpose(0, 1);

			X = X / X.norm().clamp_min(options.eps());
			auto [a, b, c] = options.ns_coefficients();

			for (int64_t i = 0; i < options.ns_steps(); ++i) {
				auto A = X.mm(X.transpose(0, 1));
				auto B = b * A + c * A.mm(A);
				X = a * X + B.mm(X);
			}

			if (transposed)
				X = X.transpose(0, 1);
			return X.to(grad.scalar_type());
		}
	};
}
