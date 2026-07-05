#pragma once
#include "Framework.h"

#include <thread_pool.h>

namespace RLGC {
	// Modified version of https://stackoverflow.com/questions/26516683/reusing-thread-in-loop-c
	struct ThreadPool {

		dp::thread_pool<>* _tp;

		ThreadPool() {
			_tp = new dp::thread_pool();
		}

		RG_NO_COPY(ThreadPool);

		~ThreadPool() {
			delete _tp;
		}

		template <typename Function, typename... Args> requires std::invocable<Function, Args...>
		void StartJobAsync(Function&& func, Args &&...args) {
			_tp->enqueue_detach(func, args...);
		}

		void StartBatchedJobs(std::function<void(int)> func, int num, bool async) {

			for (int i = 0; i < num; i++)
				StartJobAsync(func, i);

			if (!async)
				WaitUntilDone();
		}

		// Like StartBatchedJobs but enqueues ~4*threads chunk tasks instead of one task per
		// index, so a per-step call over hundreds/thousands of arenas pays a handful of queue
		// enqueues + semaphore signals rather than one per arena. func is still invoked once
		// per index (in ascending order within each chunk), so callers see identical semantics.
		void StartBatchedJobsChunked(const std::function<void(int)>& func, int num, bool async) {
			if (num <= 0) {
				if (!async)
					WaitUntilDone();
				return;
			}

			int numChunks = RS_MIN(num, RS_MAX(1, GetNumThreads() * 4));
			int chunkSize = (num + numChunks - 1) / numChunks;

			for (int c = 0; c < numChunks; c++) {
				int start = c * chunkSize;
				int stop = RS_MIN(start + chunkSize, num);
				if (start >= stop)
					break;
				StartJobAsync([func, start, stop]() {
					for (int i = start; i < stop; i++)
						func(i);
				});
			}

			if (!async)
				WaitUntilDone();
		}

		void WaitUntilDone() {
			_tp->wait_for_tasks();
		}

		int GetNumThreads() const {
			return _tp->size();
		}
	};

	extern ThreadPool g_ThreadPool;
}