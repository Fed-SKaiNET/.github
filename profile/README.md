# SKaiNET Federated Learning 

**Proposal to add privacy-preserving, decentralized training to the SKaiNET deep-learning framework.**

The project enables many devices (phones, desktops, edge nodes) to train a shared model locally, using SKaiNET’s existing on-device training APIs, while keeping all raw user data on the device. Only model updates (such as weights, gradients, and basic metrics) are sent to a central coordinator, which aggregates them (initially using FedAvg) to produce a new global model.

### The project focuses on:

* On-device / edge AI
* Federated learning with strong privacy guarantees
* Kotlin Multiplatform support (JVM, Android, etc.)
* A modular architecture with a portable federated core, pluggable networking, and a JVM reference coordinator

> [!IMPORTANT]
> **About the name**
>
> “SKaiNET” is a **working project name** chosen early in the project’s life as part of a personal learning and experimentation effort, before any trademark considerations were known.
>
> The name is **not intended to reference, infringe, or imply association with any existing trademarks, companies, or products**. It is not a commercial brand and is **not claimed or assignable** to any company or organization that contributors may be affiliated with.
>
> If a naming conflict arises, the project name may be changed in the future.


## Vision  
The goal is to make federated learning a natural extension of SKaiNET’s device-first philosophy, enabling collaborative model training without centralizing user data.

## Releases

Follow changelog tracking the current releases:

https://github.com/Fed-SKaiNET/skainet-fed/blob/develop/CHANGELOG.md

## What SKaiNET Enables for Federated Learning

### Cross-Platform Tensor Math — One Codebase, Every Device

SKaiNET's `ExecutionContext` and `TensorOps` provide a single API for tensor operations (`add`, `subtract`, `mulScalar`, `divide`, `softmax`, `sigmoid`, etc.) that works identically across JVM, Android, iOS, JS, WASM, and native Linux/macOS. This means the entire FedAvg aggregation, gradient accumulation, and optimizer logic is written once in `commonMain` and runs everywhere.

### Typed Model Parameter Access

SKaiNET's `Module<FP32, Float>` and `ModuleNode` hierarchy lets federated code extract, iterate, and update named model parameters recursively. This is the foundation for:

- **`ParameterManager`** — extracting global weights from a model
- **Weight delta computation** — `ops.subtract(current, initial)` per parameter after local training
- **Applying aggregated updates** — writing server-aggregated weights back into client modules

### Weighted Averaging on Real Tensors

FedAvg's core operation — weighted average of client weight updates — is built directly on `ops.mulScalar()` and `ops.add()`. Without SKaiNET providing element-wise tensor math, this would require raw array manipulation per platform.

### Forward Pass and Loss Computation On-Device

`module.forward(input, ctx)` combined with `FederatedLoss` implementations (MSE, CrossEntropy, BinaryCE) using SKaiNET ops (`softmax`, `sigmoid`, `mean`, `sum`) makes local on-device training possible. Clients can train, compute loss, and return results — all through SKaiNET's module system.

### Optimizers Built on Tensor Primitives

`FederatedSGD` and `FederatedAdam` are implemented entirely with SKaiNET ops — momentum updates, squared gradient tracking, bias correction, weight decay — all via `ops.add`, `ops.multiply`, `ops.mulScalar`, `ops.divide`. No custom math kernels needed.

### Memory-Efficient Training on Edge Devices

SKaiNET's `ExecutionContext` enables:

- **`TensorPool`** — reuses tensor allocations via `ctx.zeros()` and `ctx.full()` to reduce GC pressure on mobile
- **`CheckpointingGradientTape`** — wraps the execution context to trade compute for memory during backward passes
- **`GradientAccumulator`** — accumulates gradients across micro-batches using `ops.add` and `ops.mulScalar`

### Data Loading Integration

The `DataLoader<T, V>` / `DataBatch<T, V>` abstraction is generic over SKaiNET's `DType` system, connecting to `skainet-data-api` and `skainet-data-simple` for batched tensor I/O on each platform.

---

**In short:** SKaiNET provides the typed, cross-platform tensor runtime and module system that lets federated learning — aggregation, local training, optimization, and memory management — be written as pure Kotlin multiplatform code without platform-specific math or model handling.

