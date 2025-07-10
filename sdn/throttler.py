import time
from typing import Dict, Any, Union # Added Union

class Throttler:
    """
    Simulates an SDN-based traffic throttler.
    In a real-world scenario, this class would interact with an SDN controller's API
    (e.g., OpenFlow, ONOS, Ryu) to apply traffic shaping rules,
    rate limits, or prioritize certain traffic flows.

    For AutoPaaS-X, this can be used to control traffic to serverless functions
    or tenants based on detected "noisy" behavior.
    """

    def __init__(self, default_rate_limit_mbps: int = 1000):
        """
        Initializes the Throttler with a default rate limit.

        Args:
            default_rate_limit_mbps (int): The default network rate limit in Mbps,
                                           used if a specific throttle rate isn't provided.
        """
        self.current_rate_limit_mbps = default_rate_limit_mbps
        self.throttling_rules: Dict[str, Dict[str, Any]] = {} # Stores active throttling rules (tenant_id -> rule_details)
        print(f"Throttler initialized with default rate limit: {self.current_rate_limit_mbps} Mbps")

    def apply_rate_limit(self, tenant_id: str, rate_limit_mbps: int, priority: int = 5):
        """
        Applies a rate limit to a specific network flow or tenant.
        This simulates configuring flow rules on an SDN controller.

        Args:
            tenant_id (str): A unique identifier for the tenant or network flow.
            rate_limit_mbps (int): The desired rate limit for this tenant in Mbps.
            priority (int): Priority of this rule (higher value means higher precedence).
        """
        if rate_limit_mbps < 0:
            print(f"Error: Rate limit for tenant '{tenant_id}' cannot be negative. Skipping.")
            return

        self.throttling_rules[tenant_id] = {
            "rate_limit_mbps": rate_limit_mbps,
            "priority": priority,
            "timestamp": time.time()
        }
        print(f"Throttler: Applied rate limit of {rate_limit_mbps} Mbps to tenant '{tenant_id}' (Priority: {priority}).")

    def remove_rate_limit(self, tenant_id: str):
        """
        Removes a previously applied rate limit for a specific tenant.

        Args:
            tenant_id (str): The identifier of the tenant.
        """
        if tenant_id in self.throttling_rules:
            del self.throttling_rules[tenant_id]
            print(f"Throttler: Removed rate limit for tenant '{tenant_id}'.")
        else:
            print(f"Throttler: Warning: No rate limit found for tenant '{tenant_id}'.")

    def get_flow_status(self, tenant_id: str) -> Union[Dict[str, Any], None]: # Corrected type hint
        """
        Retrieves the current throttling status for a given tenant.

        Args:
            tenant_id (str): The identifier of the tenant.

        Returns:
            Union[Dict[str, Any], None]: A dictionary with rule details if found, otherwise None.
        """
        return self.throttling_rules.get(tenant_id)

    def get_all_throttling_rules(self) -> Dict[str, Any]:
        """
        Returns all currently active throttling rules.

        Returns:
            Dict[str, Any]: A dictionary of all active throttling rules.
        """
        return self.throttling_rules

    def simulate_traffic(self, tenant_id: str, data_size_mb: float):
        """
        Simulates sending data through a throttled flow and estimates time taken.
        This is a conceptual simulation and does not involve actual network traffic.

        Args:
            tenant_id (str): The identifier of the tenant.
            data_size_mb (float): The size of data to send in MB.

        Returns:
            float: Estimated time taken in seconds, or -1 if flow not found.
        """
        flow_rule = self.throttling_rules.get(tenant_id)
        if not flow_rule:
            print(f"Throttler: Error: Tenant '{tenant_id}' not found. Cannot simulate traffic.")
            return -1

        rate_mbps = flow_rule["rate_limit_mbps"]
        if rate_mbps <= 0:
            print(f"Throttler: Warning: Tenant '{tenant_id}' has a rate limit of 0 Mbps. Data will not pass.")
            return float('inf') # Infinite time

        # Convert Mbps to MBps (1 Byte = 8 bits, so 1 MBps = 8 Mbps)
        rate_mb_per_sec = rate_mbps / 8.0
        if rate_mb_per_sec == 0:
            return float('inf')

        estimated_time_sec = data_size_mb / rate_mb_per_sec
        print(f"Throttler: Simulating {data_size_mb:.2f} MB data for tenant '{tenant_id}' with rate {rate_mbps} Mbps. "
              f"Estimated time: {estimated_time_sec:.2f} seconds.")
        return estimated_time_sec

# Example Usage (for testing purposes)
if __name__ == "__main__":
    throttler = Throttler()

    print("\n--- Applying QoS rules ---")
    throttler.apply_rate_limit("tenant_A", 500) # Limit Tenant A to 500 Mbps
    throttler.apply_rate_limit("tenant_B", 1000, priority=200) # Limit Tenant B to 1000 Mbps, higher priority

    print("\n--- Checking rule status ---")
    print(f"Tenant A rule: {throttler.get_flow_status('tenant_A')}")
    print(f"Tenant C rule: {throttler.get_flow_status('tenant_C')}") # Non-existent tenant

    print("\n--- All active QoS rules ---")
    print(throttler.get_all_throttling_rules())

    print("\n--- Updating a QoS rule ---")
    throttler.apply_rate_limit("tenant_A", 200) # Update Tenant A to 200 Mbps
    print(f"Updated Tenant A rule: {throttler.get_flow_status('tenant_A')}")

    print("\n--- Simulating traffic ---")
    throttler.simulate_traffic("tenant_A", 20) # 20 MB data for Tenant A
    throttler.simulate_traffic("tenant_B", 50) # 50 MB data for Tenant B
    throttler.simulate_traffic("tenant_C", 10) # Simulate for non-existent tenant

    print("\n--- Removing a QoS rule ---")
    throttler.remove_rate_limit("tenant_B")
    print(f"Tenant B rule after removal: {throttler.get_flow_status('tenant_B')}")

    print("\n--- All active QoS rules after removal ---")
    print(throttler.get_all_throttling_rules())
