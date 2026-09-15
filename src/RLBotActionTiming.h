#pragma once

// Packet-clock scheduling, independent of inference and the RLBot transport.
struct RLBotActionTiming {
	int ticksSinceDecision = -1;

	// Infer on the first usable packet and on the packet reaching the boundary.
	// If packets were dropped, infer once from the available state and start a
	// fresh window; a new action must not inherit time before it was inferred.
	bool Advance(int ticksElapsed, int tickSkip) {
		if (ticksSinceDecision < 0) {
			ticksSinceDecision = 0;
			return true;
		}
		if (ticksElapsed > 0)
			ticksSinceDecision += ticksElapsed;
		if (ticksSinceDecision >= tickSkip) {
			ticksSinceDecision = 0;
			return true;
		}
		return false;
	}

	// Called AFTER inference: zero delay sends that decision in this callback.
	// This is a software delay only; transport/game latency is additional.
	bool ShouldApply(int actionDelay) const {
		return ticksSinceDecision >= 0 && ticksSinceDecision >= actionDelay;
	}
};
