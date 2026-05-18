# Java-side validation tests

The validator now lives next to the production code at
[`src/main/java/sim/ValidationRunner.java`](../src/main/java/sim/ValidationRunner.java),
so it gets shaded into `simulation.jar` automatically. There is no separate
test directory layout.

## Run

```powershell
# 1. Rebuild the image so the new class is in the jar.
docker compose build cloudsim-java

# 2. Run the validator. --entrypoint java overrides the default
#    "java -jar simulation.jar" entrypoint; --no-deps skips starting rl-agent.
#    Trace and results volumes are mounted automatically.
docker compose run --rm --no-deps --entrypoint java cloudsim-java `
    -cp simulation.jar sim.ValidationRunner
```

Exit code 0 = all assertions passed. Output prints `[PASS]` / `[FAIL]`
per assertion; see the source for what each one verifies.

## What it asserts

Behaviour after the B2/B3 fix (see
[`assets/report/java-validation-report.md`](../../assets/report/java-validation-report.md)):

| Test method | Status | Asserts |
|---|---|---|
| `testB1_K8sAlwaysPicksLowestIndex` | fix verified | K8s picks a different host once host 0 is loaded |
| `testB2_ResourcesNeverReleased` | fix verified | PE total ≤ cluster capacity, releases observed |
| `testB3_EnergyIntegratesLogicalTime` | fix verified | `idle×T ≤ kwh ≤ max×T` |
| `testB4_SlaMathCannotTrigger` | fix verified | pumping load to host 0 produces violations; spreading produces fewer |
| `testB5_AvgCpuUtilFormula` | partial fix | `clusterCpuUtil ∈ [0, 1]` (time-weighting still off, deferred) |
| `testB6_ClampInconsistency` | fix verified | both formulas clamp to ≤ 1 |
| `testB7_HighFilterIsNoOp` | bug still present (deferred) | HIGH = all non-Pending tasks |
| `testB8_CustomPolicyNotUsedByDatacenter` | bug still present (deferred) | DC default policy = `VmAllocationPolicySimple` |
| `testB9_DefaultConstructorSeedBug` | bug still present (deferred) | default ctor passes `pesCount` (64) as seed |
| `testB10_ActionMaskWhenDone` | bug still present (deferred) | mask is all-false when `episodeDone` |
