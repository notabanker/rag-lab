# Distributed Systems Design

## Introduction

A distributed system is a collection of independent computers that appears to its
users as a single coherent system. Distributed systems are characterized by
concurrency of components, lack of a global clock, and independent failure of
components.

## Chapter 1: Consensus Protocols

Consensus is the process by which distributed nodes agree on a single data value.
The two most influential consensus algorithms are Paxos and Raft.

Paxos, developed by Leslie Lamport, works through a multi-phase protocol:
- Phase 1 (Prepare): A proposer selects a proposal number and sends prepare
  requests to a majority of acceptors.
- Phase 2 (Accept): If the proposer receives promises from a majority, it sends
  accept requests with its proposed value.
- The value is chosen when a majority of acceptors accept it.

Raft was designed as a more understandable alternative to Paxos. It separates
consensus into leader election, log replication, and safety. Raft uses a strong
leader model where the leader manages the replicated log.

## Chapter 2: CAP Theorem

The CAP theorem states that a distributed system can provide at most two of
three guarantees:
- Consistency: All nodes see the same data at the same time
- Availability: Every request receives a non-error response
- Partition tolerance: The system continues to operate despite network partitions

In practice, network partitions are unavoidable, so systems must choose between
consistency (CP systems like etcd) and availability (AP systems like Cassandra).

## Chapter 3: Failure Detection

Failure detectors are a fundamental building block of fault-tolerant distributed
systems. An eventually perfect failure detector (<>P) eventually suspects every
crashed process and eventually trusts every correct process.

Common failure detection mechanisms:
- Heartbeat-based: Nodes send periodic heartbeats; missing heartbeats indicate failure
- Gossip-based: Nodes share membership information probabilistically
- Accrual failure detectors: Return a suspicion level rather than binary up/down

## Chapter 4: Distributed Transactions

Two-Phase Commit (2PC) is the classic distributed transaction protocol:
- Phase 1 (Voting): Coordinator asks all participants to prepare
- Phase 2 (Commit): If all vote yes, coordinator commits; otherwise aborts

2PC has a blocking problem: if the coordinator fails after sending prepare but
before sending commit, participants are blocked.

Three-Phase Commit (3PC) adds a pre-commit phase to reduce blocking, but is
rarely used in practice due to its overhead and corner cases.

The SAGA pattern provides an alternative for long-lived transactions by breaking
them into a sequence of local transactions with compensating actions for rollback.
