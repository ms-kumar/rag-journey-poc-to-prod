"""
Demo: Cost tracking and model selection system.

Demonstrates:
1. Cost tracking per model
2. Quality/latency/cost matrix
3. Tiered model selection
4. Autoscaling policies
5. Cost per 1k queries calculation
"""

import random
import time

from src.services.cost import (
    AutoscalingPolicy,
    Autoscaler,
    CostTracker,
    LoadMetrics,
    ModelSelector,
    ModelTier,
    ScalingDecision,
    SelectionStrategy,
)


def demo_cost_tracking():
    """Demo cost tracking functionality."""
    print("\n" + "=" * 80)
    print("DEMO 1: COST TRACKING")
    print("=" * 80)
    
    tracker = CostTracker()
    
    # Simulate requests to different models
    models_used = [
        ("gpt2", 0.0001, 0.05),  # (model, cost, latency)
        ("gpt-3.5-turbo", 0.002, 0.5),
        ("gpt-4", 0.03, 2.0),
        ("flan-t5-base", 0.0005, 0.1),
    ]
    
    print("\nSimulating 1000 requests across different models...")
    
    for i in range(1000):
        model, cost, latency = random.choice(models_used)
        quality = random.uniform(0.7, 0.95)
        tokens = random.randint(100, 500)
        error = random.random() < 0.02  # 2% error rate
        
        tracker.record_request(
            model_name=model,
            cost=cost,
            latency=latency,
            tokens=tokens,
            quality_score=quality,
            error=error,
        )
    
    # Generate report
    tracker.print_summary()
    
    # Show efficiency rankings
    print("\nModel Efficiency Rankings (Quality/Cost):")
    print("-" * 60)
    for name, efficiency in tracker.get_models_by_efficiency():
        print(f"  {name}: {efficiency:.2f}")
    
    return tracker


def demo_model_selection():
    """Demo tiered model selection."""
    print("\n" + "=" * 80)
    print("DEMO 2: TIERED MODEL SELECTION")
    print("=" * 80)
    
    selector = ModelSelector()
    
    # Show comparison matrix
    selector.print_comparison_matrix()
    
    print("\n" + "=" * 80)
    print("SELECTION SCENARIOS")
    print("=" * 80)
    
    # Scenario 1: Minimize cost
    print("\n1. Minimize Cost (min quality 0.6):")
    model = selector.select_model(
        strategy=SelectionStrategy.MINIMIZE_COST,
        min_quality=0.6,
    )
    print(f"   -> Selected: {model.name} (${model.cost_per_1k:.2f}/1k)")
    
    # Scenario 2: Minimize latency
    print("\n2. Minimize Latency (min quality 0.7):")
    model = selector.select_model(
        strategy=SelectionStrategy.MINIMIZE_LATENCY,
        min_quality=0.7,
    )
    print(f"   -> Selected: {model.name} ({model.avg_latency_ms:.0f}ms)")
    
    # Scenario 3: Maximize quality
    print("\n3. Maximize Quality (max cost $5/1k):")
    model = selector.select_model(
        strategy=SelectionStrategy.MAXIMIZE_QUALITY,
        max_cost_per_1k=5.0,
    )
    print(f"   -> Selected: {model.name} (quality: {model.quality_score:.2f})")
    
    # Scenario 4: Balanced selection
    print("\n4. Balanced (optimize efficiency):")
    model = selector.select_model(
        strategy=SelectionStrategy.BALANCED,
        min_quality=0.75,
    )
    print(
        f"   -> Selected: {model.name} "
        f"(efficiency: {model.efficiency_score:.2f})"
    )
    
    # Scenario 5: Budget-constrained
    print("\n5. Budget-Constrained ($100 budget):")
    selector.set_budget(100.0)
    
    for i in range(3):
        model = selector.select_model(
            strategy=SelectionStrategy.ADAPTIVE,
            min_quality=0.7,
        )
        cost = model.cost_per_1k / 1000 * 100  # Simulate 100 requests
        selector.record_cost(cost)
        print(
            f"   Iteration {i+1}: {model.name} "
            f"(cost: ${cost:.2f}, remaining: ${selector.remaining_budget:.2f})"
        )
    
    # Scenario 6: Tier-specific selection
    print("\n6. Premium Tier Only:")
    model = selector.select_model(
        strategy=SelectionStrategy.BALANCED,
        tier=ModelTier.PREMIUM,
    )
    print(f"   -> Selected: {model.name} ({model.tier.value} tier)")
    
    return selector


def demo_autoscaling():
    """Demo autoscaling functionality."""
    print("\n" + "=" * 80)
    print("DEMO 3: AUTOSCALING")
    print("=" * 80)
    
    # Create autoscaling policy
    policy = AutoscalingPolicy(
        min_instances=1,
        max_instances=10,
        scale_up_cpu_threshold=70.0,
        scale_down_cpu_threshold=30.0,
        scale_up_queue_threshold=50,
        cooldown_period=10,  # Short cooldown for demo
        max_cost_per_hour=50.0,
    )
    
    autoscaler = Autoscaler(policy)
    autoscaler.print_status()
    
    # Simulate load scenarios
    scenarios = [
        ("Low Load", 25, 5, 10, 200, 300, 1.0, 50),
        ("Medium Load", 65, 30, 20, 400, 500, 2.0, 150),
        ("High Load", 85, 80, 40, 800, 1200, 3.0, 300),
        ("Spike Load", 95, 120, 60, 1500, 2500, 5.0, 500),
        ("Cooling Down", 60, 40, 25, 500, 700, 2.5, 200),
        ("Low Load Again", 20, 3, 8, 150, 250, 0.8, 40),
    ]
    
    print("\n" + "=" * 80)
    print("LOAD SIMULATION")
    print("=" * 80)
    
    for scenario_name, cpu, queue, active, avg_lat, p95_lat, rps, err_rate in scenarios:
        print(f"\n{scenario_name}:")
        print(
            f"  CPU: {cpu}%, Queue: {queue}, "
            f"Latency: {avg_lat}/{p95_lat}ms (avg/p95)"
        )
        
        metrics = LoadMetrics(
            cpu_usage=cpu,
            queue_size=queue,
            active_requests=active,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95_lat,
            error_rate=err_rate,
            requests_per_second=rps,
        )
        
        decision, new_instances = autoscaler.auto_scale(metrics)
        
        if decision != ScalingDecision.NO_CHANGE:
            print(f"  -> {decision.value.upper()}: {new_instances} instances")
        else:
            print(f"  -> No scaling needed ({new_instances} instances)")
        
        # Simulate cost
        cost = new_instances * 5.0  # $5 per instance per period
        autoscaler.record_cost(cost)
        
        time.sleep(0.5)  # Brief pause for demo
    
    # Show final status
    autoscaler.print_status()
    
    return autoscaler


def demo_integrated_system():
    """Demo integrated cost tracking + model selection + autoscaling."""
    print("\n" + "=" * 80)
    print("DEMO 4: INTEGRATED SYSTEM")
    print("=" * 80)
    
    tracker = CostTracker()
    selector = ModelSelector()
    
    policy = AutoscalingPolicy(
        min_instances=1,
        max_instances=8,
        max_cost_per_hour=100.0,
    )
    autoscaler = Autoscaler(policy)
    
    # Set budget
    budget = 50.0
    selector.set_budget(budget)
    print(f"\nStarting with ${budget:.2f} budget")
    
    # Simulate varying load
    print("\nSimulating 500 requests with varying load...")
    
    for i in range(500):
        # Determine load level
        if i < 100:
            load = "low"
            cpu = random.uniform(20, 40)
            queue = random.randint(0, 10)
        elif i < 300:
            load = "medium"
            cpu = random.uniform(50, 70)
            queue = random.randint(10, 40)
        else:
            load = "high"
            cpu = random.uniform(75, 95)
            queue = random.randint(40, 100)
        
        # Select model based on load
        if load == "low":
            strategy = SelectionStrategy.MINIMIZE_COST
        elif load == "medium":
            strategy = SelectionStrategy.BALANCED
        else:
            strategy = SelectionStrategy.MINIMIZE_LATENCY
        
        try:
            model = selector.select_model(
                strategy=strategy,
                min_quality=0.6,
            )
            
            # Simulate request
            cost = model.cost_per_1k / 1000
            latency = model.avg_latency_ms / 1000
            quality = model.quality_score + random.uniform(-0.05, 0.05)
            
            # Record in tracker
            tracker.record_request(
                model_name=model.name,
                cost=cost,
                latency=latency,
                tokens=random.randint(100, 500),
                quality_score=quality,
            )
            
            # Update selector budget
            selector.record_cost(cost)
            
            # Check autoscaling every 50 requests
            if i % 50 == 0:
                metrics = LoadMetrics(
                    cpu_usage=cpu,
                    queue_size=queue,
                    active_requests=random.randint(5, 30),
                    avg_latency_ms=latency * 1000,
                    p95_latency_ms=latency * 1200,
                    error_rate=1.0,
                    requests_per_second=10.0,
                )
                
                decision, instances = autoscaler.auto_scale(metrics)
                
                if decision != ScalingDecision.NO_CHANGE:
                    print(
                        f"\n[Request {i}] Load: {load}, "
                        f"Scaling: {decision.value} -> {instances} instances"
                    )
        
        except ValueError as e:
            print(f"\n[Request {i}] Selection failed: {e}")
            break
    
    # Show results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    report = tracker.get_report()
    print(f"\nTotal Cost: ${report.total_cost:.2f}")
    print(f"Budget Remaining: ${selector.remaining_budget:.2f}")
    print(f"Total Requests: {report.total_requests}")
    print(f"Cost per 1k queries: ${report.cost_per_1k:.2f}")
    
    print("\nTop Models by Usage:")
    for name, cost in tracker.get_top_models_by_cost(top_n=3):
        metrics = tracker.get_model_metrics(name)
        if metrics:
            print(
                f"  {name}: {metrics.total_requests} requests, "
                f"${cost:.2f} total, ${metrics.cost_per_1k:.2f}/1k"
            )
    
    autoscaler.print_status()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("COST TRACKING & MODEL SELECTION DEMO")
    print("=" * 80)
    
    # Run demos
    demo_cost_tracking()
    time.sleep(1)
    
    demo_model_selection()
    time.sleep(1)
    
    demo_autoscaling()
    time.sleep(1)
    
    demo_integrated_system()
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
