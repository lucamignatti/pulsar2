#!/bin/bash
# Bind this MPI local rank to the POWER9 socket that owns its GPU.
# dcs compute: GPU0-2 NUMA 0 CPUs 0-79; GPU3-5 NUMA 8 CPUs 80-159.
# Spectrum MPI default is map-by/bind-to socket with cyclic ranking, which
# put rank 1 on NUMA 8 with GPU 1 on NUMA 0 (and rank 4 the reverse).
set -e
r="${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_RANK:-0}}"
case "$r" in
	0) exec numactl --physcpubind=0-26 --localalloc -- ./build/GigaLearnBot "$@" ;;
	1) exec numactl --physcpubind=27-53 --localalloc -- ./build/GigaLearnBot "$@" ;;
	2) exec numactl --physcpubind=54-79 --localalloc -- ./build/GigaLearnBot "$@" ;;
	3) exec numactl --physcpubind=80-106 --localalloc -- ./build/GigaLearnBot "$@" ;;
	4) exec numactl --physcpubind=107-133 --localalloc -- ./build/GigaLearnBot "$@" ;;
	5) exec numactl --physcpubind=134-159 --localalloc -- ./build/GigaLearnBot "$@" ;;
	*)
		echo "pulsar2_numa_rank.sh: unexpected local rank '$r'" >&2
		exec ./build/GigaLearnBot "$@"
		;;
esac
