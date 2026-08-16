#include "Fragment.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace GGL {

namespace {
constexpr int8_t kTruncTerm = 2;

void AppendBytes(std::vector<uint8_t>& out, const void* p, size_t n) {
	const uint8_t* b = static_cast<const uint8_t*>(p);
	out.insert(out.end(), b, b + n);
}
void AppendTensor(std::vector<uint8_t>& out, const torch::Tensor& t) {
	if (!t.defined() || t.numel() == 0)
		return;
	auto c = t.contiguous().cpu();
	AppendBytes(out, c.data_ptr(), (size_t)c.numel() * c.element_size());
}
torch::Tensor Take(const uint8_t*& p, const uint8_t* end, at::IntArrayRef sizes, torch::ScalarType dt) {
	int64_t n = 1;
	for (auto s : sizes)
		n *= s;
	if (n == 0)
		return torch::empty(sizes, torch::TensorOptions().dtype(dt));
	size_t bytes = (size_t)n * torch::elementSize(dt);
	if (p + bytes > end)
		throw std::runtime_error("Fragment unpack overrun");
	auto t = torch::empty(sizes, torch::TensorOptions().dtype(dt).pinned_memory(true));
	std::memcpy(t.data_ptr(), p, bytes);
	p += bytes;
	return t;
}

std::vector<int64_t> TruncRowIdx(const torch::Tensor& terminals) {
	std::vector<int64_t> out;
	if (!terminals.defined() || terminals.numel() == 0)
		return out;
	auto t = terminals.contiguous();
	const int8_t* p = t.data_ptr<int8_t>();
	for (int64_t i = 0; i < t.size(0); i++)
		if (p[i] == kTruncTerm)
			out.push_back(i);
	return out;
}

torch::Tensor GatherNext(const torch::Tensor& ns, const std::vector<int64_t>& rows, int obsSize) {
	if (rows.empty())
		return torch::empty({ 0, obsSize }, torch::kFloat32);
	if (!ns.defined() || ns.size(0) == 0)
		return torch::empty({ (int64_t)rows.size(), obsSize }, torch::kFloat32);
	std::vector<torch::Tensor> parts;
	parts.reserve(rows.size());
	for (int64_t r : rows) {
		if (r >= 0 && r < ns.size(0))
			parts.push_back(ns.narrow(0, r, 1));
	}
	if (parts.empty())
		return torch::empty({ 0, obsSize }, torch::kFloat32);
	return torch::cat(parts, 0).contiguous();
}

torch::Tensor NarrowOpt(const torch::Tensor& t, int64_t start, int64_t len) {
	if (!t.defined() || t.numel() == 0)
		return t;
	return t.narrow(0, start, len).contiguous();
}

void RecomputeVersions(TrajectoryFragment& f) {
	if (!f.policy_version.defined() || f.policy_version.numel() == 0)
		return;
	auto t = f.policy_version.contiguous();
	const int32_t* p = t.data_ptr<int32_t>();
	int32_t mn = std::numeric_limits<int32_t>::max();
	int32_t mx = 0;
	for (int64_t i = 0; i < t.size(0); i++) {
		mn = std::min(mn, p[i]);
		mx = std::max(mx, p[i]);
	}
	f.hdr.min_policy_version = (uint64_t)(uint32_t)mn;
	f.hdr.max_policy_version = (uint64_t)(uint32_t)mx;
}

void SliceColumns(TrajectoryFragment& o, const TrajectoryFragment& src, int64_t start, int64_t len) {
	o.states = src.states.narrow(0, start, len).contiguous();
	o.actions = src.actions.narrow(0, start, len).contiguous();
	o.logProbs = src.logProbs.narrow(0, start, len).contiguous();
	o.rewards = src.rewards.narrow(0, start, len).contiguous();
	o.terminals = src.terminals.narrow(0, start, len).contiguous();
	o.actionMasks = src.actionMasks.narrow(0, start, len).contiguous();
	o.policy_version = src.policy_version.narrow(0, start, len).contiguous();
	o.goalRews = NarrowOpt(src.goalRews, start, len);
	o.carHerGoals = NarrowOpt(src.carHerGoals, start, len);
	o.ballHerGoals = NarrowOpt(src.ballHerGoals, start, len);
	o.ballMovedMask = NarrowOpt(src.ballMovedMask, start, len);
	o.carStateHerGoals = NarrowOpt(src.carStateHerGoals, start, len);
	o.gatedPos = NarrowOpt(src.gatedPos, start, len);
	o.touched = NarrowOpt(src.touched, start, len);
	o.oppTouched = NarrowOpt(src.oppTouched, start, len);
	o.oppStates = NarrowOpt(src.oppStates, start, len);
	o.oppActionMasks = NarrowOpt(src.oppActionMasks, start, len);
	o.practiceMask = NarrowOpt(src.practiceMask, start, len);
}
}

std::vector<uint8_t> TrajectoryFragment::Pack() const {
	std::vector<uint8_t> out;
	out.reserve(1 << 20);
	AppendBytes(out, &hdr, sizeof(hdr));
	AppendTensor(out, states);
	AppendTensor(out, actions);
	AppendTensor(out, logProbs);
	AppendTensor(out, rewards);
	AppendTensor(out, terminals);
	AppendTensor(out, actionMasks);
	AppendTensor(out, policy_version);
	AppendTensor(out, nextStates);
	if (hdr.flags & 1) AppendTensor(out, goalRews);
	if (hdr.flags & 2) {
		AppendTensor(out, carHerGoals);
		AppendTensor(out, ballHerGoals);
		AppendTensor(out, ballMovedMask);
		AppendTensor(out, gatedPos);
		AppendTensor(out, touched);
		AppendTensor(out, oppTouched);
		AppendTensor(out, oppStates);
		AppendTensor(out, oppActionMasks);
	}
	if (hdr.flags & 4) AppendTensor(out, carStateHerGoals);
	if (hdr.flags & 16) AppendTensor(out, practiceMask);
	return out;
}

TrajectoryFragment TrajectoryFragment::Unpack(const uint8_t* data, size_t nbytes) {
	TrajectoryFragment f;
	if (nbytes < sizeof(FragmentHeader))
		throw std::runtime_error("Fragment too small");
	std::memcpy(&f.hdr, data, sizeof(FragmentHeader));
	if (f.hdr.magic != kFragmentMagic)
		throw std::runtime_error("Fragment magic");
	const uint8_t* p = data + sizeof(FragmentHeader);
	const uint8_t* end = data + nbytes;
	const int64_t T = f.hdr.T;
	const int64_t o = f.hdr.obsSize;
	const int64_t a = f.hdr.numActions;
	const int64_t nTrunc = f.hdr.nTrunc;
	f.states = Take(p, end, { T, o }, torch::kFloat32);
	f.actions = Take(p, end, { T }, torch::kInt32);
	f.logProbs = Take(p, end, { T }, torch::kFloat32);
	f.rewards = Take(p, end, { T }, torch::kFloat32);
	f.terminals = Take(p, end, { T }, torch::kInt8);
	f.actionMasks = Take(p, end, { T, a }, torch::kUInt8);
	f.policy_version = Take(p, end, { T }, torch::kInt32);
	f.nextStates = Take(p, end, { nTrunc, o }, torch::kFloat32);
	if (f.hdr.flags & 1)
		f.goalRews = Take(p, end, { T }, torch::kFloat32);
	if (f.hdr.flags & 2) {
		f.carHerGoals = Take(p, end, { T, 6 }, torch::kFloat32);
		f.ballHerGoals = Take(p, end, { T, 6 }, torch::kFloat32);
		f.ballMovedMask = Take(p, end, { T }, torch::kUInt8);
		f.gatedPos = Take(p, end, { T }, torch::kFloat32);
		f.touched = Take(p, end, { T }, torch::kUInt8);
		f.oppTouched = Take(p, end, { T }, torch::kUInt8);
		f.oppStates = Take(p, end, { T, o }, torch::kFloat32);
		f.oppActionMasks = Take(p, end, { T, a }, torch::kUInt8);
	}
	if (f.hdr.flags & 4)
		f.carStateHerGoals = Take(p, end, { T, 6 }, torch::kFloat32);
	if (f.hdr.flags & 16 && p < end)
		f.practiceMask = Take(p, end, { T }, torch::kUInt8);
	return f;
}

TrajectoryFragment TrajectoryFragment::SlicePrefix(int64_t k) const {
	TrajectoryFragment p = *this;
	const auto truncs = TruncRowIdx(terminals);
	SliceColumns(p, *this, 0, k);
	auto term = p.terminals.clone();
	int8_t* tp = term.data_ptr<int8_t>();
	const bool already = tp[k - 1] == kTruncTerm;
	tp[k - 1] = kTruncTerm;
	p.terminals = term;

	std::vector<int64_t> nsRows;
	for (int64_t i = 0; i < (int64_t)truncs.size(); i++) {
		if (truncs[i] < k)
			nsRows.push_back(i);
	}
	p.nextStates = GatherNext(nextStates, nsRows, hdr.obsSize);
	if (!already) {
		auto extra = states.narrow(0, k, 1).contiguous();
		if (p.nextStates.size(0) == 0)
			p.nextStates = extra;
		else
			p.nextStates = torch::cat({ p.nextStates, extra }, 0).contiguous();
	}
	p.hdr.T = (int32_t)k;
	p.hdr.nTrunc = (int32_t)p.nextStates.size(0);
	p.hdr.env_steps = (uint64_t)k;
	RecomputeVersions(p);
	return p;
}

TrajectoryFragment TrajectoryFragment::SliceSuffix(int64_t k) const {
	TrajectoryFragment s = *this;
	const int64_t T = hdr.T;
	const auto truncs = TruncRowIdx(terminals);
	SliceColumns(s, *this, k, T - k);
	std::vector<int64_t> nsRows;
	for (int64_t i = 0; i < (int64_t)truncs.size(); i++) {
		if (truncs[i] >= k)
			nsRows.push_back(i);
	}
	s.nextStates = GatherNext(nextStates, nsRows, hdr.obsSize);
	s.hdr.T = (int32_t)(T - k);
	s.hdr.nTrunc = (int32_t)s.nextStates.size(0);
	s.hdr.env_steps = (uint64_t)s.hdr.T;
	RecomputeVersions(s);
	return s;
}

}
