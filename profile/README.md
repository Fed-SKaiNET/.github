# SKaiNET Federated Learning 

**Proposal to add privacy-preserving, decentralized training to the SKaiNET deep-learning framework.**

The project enables many devices (phones, desktops, edge nodes) to train a shared model locally, using SKaiNET’s existing on-device training APIs, while keeping all raw user data on the device. Only model updates (such as weights, gradients, and basic metrics) are sent to a central coordinator, which aggregates them (initially using FedAvg) to produce a new global model.

### The project focuses on:

* On-device / edge AI
* Federated learning with strong privacy guarantees
* Kotlin Multiplatform support (JVM, Android, etc.)
* A modular architecture with a portable federated core, pluggable networking, and a JVM reference coordinator

## Vision  
The goal is to make federated learning a natural extension of SKaiNET’s device-first philosophy, enabling collaborative model training without centralizing user data.

## Releases

Follow changelog tracking the current releases:

https://github.com/Fed-SKaiNET/skainet-fed/blob/develop/CHANGELOG.md
