#include "TestHarness.h"

#include <GigaLearnCPP/Util/MetricSink.h>
#include <GigaLearnCPP/Util/Report.h>

// MetricSink port + in-memory adapter.
//
// This is the seam that lets a Learner be built and exercised without an embedded Python
// interpreter or wandb. The test drives the in-memory adapter *through the port* so it also
// pins that virtual dispatch (and the empty run id) behaves.

using namespace GGL;

static void metric_sink_captures_through_port() {
    Report r;
    r["Policy Loss"] = 1.5;
    r["Critic Loss"] = 2.0;

    InMemoryMetricSink sink;
    MetricSink* port = &sink;  // drive it through the port, not the concrete type

    TCHECK(sink.Empty());
    port->Send(r);

    TCHECK(!sink.Empty());
    TCHECK_NEAR(sink.Last()["Policy Loss"], 1.5, 1e-9);
    TCHECK_NEAR(sink.Last()["Critic Loss"], 2.0, 1e-9);
}

static void metric_sink_accumulates_history() {
    InMemoryMetricSink sink;
    MetricSink* port = &sink;

    Report a; a["Total Iterations"] = 1;
    Report b; b["Total Iterations"] = 2;
    port->Send(a);
    port->Send(b);

    TCHECK(sink.history.size() == 2);
    TCHECK_NEAR(sink.Last()["Total Iterations"], 2.0, 1e-9);
}

static void metric_sink_has_no_run_id() {
    InMemoryMetricSink sink;
    MetricSink* port = &sink;
    // The in-memory adapter isn't a wandb run, so it reports no run id.
    TCHECK(port->GetRunID().empty());
}

void RunMetricSinkTests() {
    RUN_SUITE("metric_sink::captures_through_port", metric_sink_captures_through_port);
    RUN_SUITE("metric_sink::accumulates_history", metric_sink_accumulates_history);
    RUN_SUITE("metric_sink::has_no_run_id", metric_sink_has_no_run_id);
}
