"""
Example demonstrating retry and health check features.

This script shows how to:
1. Use retry decorators for resilient operations
2. Configure retry behavior
3. Query health check endpoints
"""

import asyncio
import time

import httpx

from src.services.retry import RetryConfig, async_retry_with_backoff, retry_with_backoff


# Example 1: Basic retry with default config
@retry_with_backoff()
def fetch_with_retry(url: str) -> dict:
    """Fetch data with automatic retry on failures."""
    response = httpx.get(url, timeout=5.0)
    response.raise_for_status()
    return response.json()


# Example 2: Custom retry configuration
custom_config = RetryConfig(
    max_retries=5,
    initial_delay=0.5,
    max_delay=10.0,
    exponential_base=2.0,
    jitter=True,
)


@retry_with_backoff(custom_config)
def fetch_with_custom_retry(url: str) -> dict:
    """Fetch with custom retry settings."""
    response = httpx.get(url, timeout=5.0)
    response.raise_for_status()
    return response.json()


# Example 3: Async retry
@async_retry_with_backoff(custom_config)
async def async_fetch_with_retry(url: str) -> dict:
    """Async fetch with retry."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=5.0)
        response.raise_for_status()
        return response.json()


# Example 4: Health check queries
async def check_api_health(base_url: str = "http://localhost:8000"):
    """Query all health check endpoints."""
    async with httpx.AsyncClient() as client:
        print("=" * 60)
        print("HEALTH CHECK EXAMPLES")
        print("=" * 60)

        # Basic health
        print("\n1. Basic Health Check (/api/v1/health)")
        print("-" * 60)
        try:
            response = await client.get(f"{base_url}/api/v1/health", timeout=5.0)
            health = response.json()
            print(f"Status: {health['status']}")
            print(f"Version: {health['version']}")
            print(f"Uptime: {health['uptime_seconds']:.2f}s")
        except Exception as e:
            print(f"Error: {e}")

        # Detailed health
        print("\n2. Detailed Health Check (with components)")
        print("-" * 60)
        try:
            response = await client.get(
                f"{base_url}/api/v1/health/detailed",
                params={"check_components": True},
                timeout=10.0,
            )
            health = response.json()
            print(f"Overall Status: {health['status']}")
            print(f"\nComponents:")
            for comp in health.get("components", []):
                print(
                    f"  - {comp['name']}: {comp['status']} "
                    f"({comp.get('response_time_ms', 0):.2f}ms)"
                )
            if "system_info" in health:
                print(f"\nSystem: {health['system_info']['platform']}")
                print(f"Python: {health['system_info']['python_version']}")
        except Exception as e:
            print(f"Error: {e}")

        # Readiness check
        print("\n3. Readiness Check (/api/v1/health/ready)")
        print("-" * 60)
        try:
            response = await client.get(f"{base_url}/api/v1/health/ready", timeout=5.0)
            ready = response.json()
            print(f"Ready: {ready.get('ready', False)}")
            print(f"Status: {ready.get('status', 'unknown')}")
        except Exception as e:
            print(f"Error: {e}")

        # Liveness check
        print("\n4. Liveness Check (/api/v1/health/live)")
        print("-" * 60)
        try:
            response = await client.get(f"{base_url}/api/v1/health/live", timeout=5.0)
            live = response.json()
            print(f"Alive: {live.get('alive', False)}")
            print(f"Uptime: {live.get('uptime_seconds', 0):.2f}s")
        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "=" * 60)


# Example 5: Retry in action
def demonstrate_retry():
    """Demonstrate retry behavior with simulated failures."""
    print("\n" + "=" * 60)
    print("RETRY DEMONSTRATION")
    print("=" * 60)

    call_count = 0

    @retry_with_backoff(RetryConfig(max_retries=3, initial_delay=0.5, jitter=False))
    def flaky_function():
        nonlocal call_count
        call_count += 1
        print(f"\nAttempt {call_count}...")

        if call_count < 3:
            print(f"  ❌ Failed (simulated transient error)")
            raise ConnectionError("Simulated failure")

        print(f"  ✅ Success!")
        return "Success after retries"

    try:
        start = time.time()
        result = flaky_function()
        elapsed = time.time() - start
        print(f"\nResult: {result}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Total attempts: {call_count}")
    except Exception as e:
        print(f"\nFailed after all retries: {e}")

    print("=" * 60)


# Main execution
async def main():
    """Run all examples."""
    print("\n🔁 RETRY & HEALTH CHECK EXAMPLES\n")

    # Demonstrate retry
    demonstrate_retry()

    # Query health checks (requires server running)
    print("\n\n💡 To test health checks, start the server:")
    print("   uv run python -m src.main")
    print("\nThen run this script again, or manually test:")
    print("   curl http://localhost:8000/api/v1/health")
    print("   curl http://localhost:8000/api/v1/health/detailed?check_components=true")
    print("   curl http://localhost:8000/api/v1/health/ready")
    print("   curl http://localhost:8000/api/v1/health/live")

    # Try to check health (may fail if server not running)
    try:
        await check_api_health()
    except Exception as e:
        print(f"\n⚠️  Could not connect to server: {e}")
        print("   Start the server to test health endpoints")


if __name__ == "__main__":
    asyncio.run(main())
