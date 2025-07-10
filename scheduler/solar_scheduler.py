import datetime
from typing import Callable, Dict, Any

class SolarScheduler:
    """
    A scheduler component that makes decisions based on simulated solar energy forecasts.
    This can be used to schedule workloads (e.g., batch jobs, less critical tasks)
    during periods of high solar energy availability to reduce operational costs
    and environmental impact.
    """

    def __init__(self, solar_prediction_func: Callable[[datetime.datetime], float] = None):
        """
        Initializes the SolarScheduler.

        Args:
            solar_prediction_func (Callable[[datetime.datetime], float], optional):
                A function that takes a datetime object and returns a float representing
                the predicted solar energy availability (e.g., in percentage or kW).
                If None, a simple dummy prediction function is used.
        """
        self.solar_prediction_func = solar_prediction_func if solar_prediction_func else self._default_solar_prediction
        self.high_solar_threshold = 70.0 # Percentage or arbitrary unit
        self.medium_solar_threshold = 40.0
        print(f"SolarScheduler initialized. High solar threshold: {self.high_solar_threshold}%")

    def _default_solar_prediction(self, dt: datetime.datetime) -> float:
        """
        A simple dummy function to simulate solar energy prediction based on time of day.
        This would be replaced by a real solar forecast API or model in a production system.

        Args:
            dt (datetime.datetime): The datetime for which to predict solar availability.

        Returns:
            float: Simulated solar availability percentage.
        """
        hour = dt.hour
        # Simulate higher solar availability during typical daylight hours
        if 8 <= hour <= 17: # 8 AM to 5 PM
            # Peak at noon (12-13), lower towards morning/evening
            if 11 <= hour <= 14: # Peak hours
                return 85.0 + (5 * (hour - 12) / 2) # Roughly 80-90%
            elif 9 <= hour <= 10 or 15 <= hour <= 16:
                return 60.0 + (10 * (hour - 9) / 1) if hour <= 10 else 60.0 + (10 * (16 - hour) / 1) # 60-70%
            else: # Early morning/late afternoon
                return 30.0 + (10 * (hour - 8) / 1) if hour == 8 else 30.0 + (10 * (17 - hour) / 1) # 30-40%
        else:
            return 10.0 # Low or no solar availability at night/very early morning

    def get_solar_forecast(self, target_datetime: datetime.datetime = None) -> float:
        """
        Retrieves the predicted solar energy availability for a given datetime.

        Args:
            target_datetime (datetime.datetime, optional): The specific datetime
                                                           for which to get the forecast.
                                                           Defaults to current UTC time.

        Returns:
            float: The predicted solar energy availability.
        """
        if target_datetime is None:
            target_datetime = datetime.datetime.utcnow() # Use UTC for consistency

        forecast = self.solar_prediction_func(target_datetime)
        print(f"SolarScheduler: Forecast for {target_datetime.isoformat()}: {forecast:.2f}% solar availability.")
        return forecast

    def decide_task_scheduling(self,
                               task_name: str,
                               current_datetime: datetime.datetime = None) -> Dict[str, Any]:
        """
        Decides whether a task should be scheduled now based on solar availability.

        Args:
            task_name (str): The name of the task to be scheduled.
            current_datetime (datetime.datetime, optional): The current datetime.
                                                            Defaults to current UTC time.

        Returns:
            Dict[str, Any]: A dictionary indicating the scheduling decision.
                            e.g., {"schedule_now": True, "reason": "High solar availability"}
        """
        if current_datetime is None:
            current_datetime = datetime.datetime.utcnow()

        solar_availability = self.get_solar_forecast(current_datetime)

        decision = {
            "task_name": task_name,
            "solar_availability": solar_availability,
            "schedule_now": False,
            "reason": "Low solar availability or off-peak hours."
        }

        if solar_availability >= self.high_solar_threshold:
            decision["schedule_now"] = True
            decision["reason"] = "High solar availability, optimal for energy-efficient tasks."
        elif solar_availability >= self.medium_solar_threshold:
            decision["schedule_now"] = True
            decision["reason"] = "Medium solar availability, suitable for non-critical tasks."
        else:
            decision["schedule_now"] = False
            decision["reason"] = "Low solar availability, consider deferring task."

        print(f"SolarScheduler: Decision for '{task_name}': {decision['schedule_now']} - {decision['reason']}")
        return decision

# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Initialize with default solar prediction
    scheduler = SolarScheduler()

    print("\n--- Testing Solar Forecasts ---")
    # Test at different times of the day (UTC)
    now = datetime.datetime.utcnow()
    morning_peak = now.replace(hour=9, minute=30, second=0, microsecond=0)
    noon_peak = now.replace(hour=12, minute=0, second=0, microsecond=0)
    evening_low = now.replace(hour=18, minute=0, second=0, microsecond=0)
    night_low = now.replace(hour=3, minute=0, second=0, microsecond=0)

    scheduler.get_solar_forecast(morning_peak)
    scheduler.get_solar_forecast(noon_peak)
    scheduler.get_solar_forecast(evening_low)
    scheduler.get_solar_forecast(night_low)
    scheduler.get_solar_forecast(now) # Current time

    print("\n--- Deciding Task Scheduling ---")
    # Simulate a critical task that needs to run regardless of solar
    print("\nCritical Task:")
    critical_task_decision = scheduler.decide_task_scheduling("critical_db_sync", night_low)
    print(f"Decision for 'critical_db_sync' at night: {critical_task_decision}")

    # Simulate a batch processing task that should prefer high solar
    print("\nBatch Processing Task (High Solar Preference):")
    batch_task_decision_noon = scheduler.decide_task_scheduling("daily_analytics_job", noon_peak)
    print(f"Decision for 'daily_analytics_job' at noon: {batch_task_decision_noon}")

    batch_task_decision_night = scheduler.decide_task_scheduling("daily_analytics_job", night_low)
    print(f"Decision for 'daily_analytics_job' at night: {batch_task_decision_night}")

    # Simulate a less critical report generation task
    print("\nReport Generation Task (Medium Solar Preference):")
    report_task_decision_morning = scheduler.decide_task_scheduling("monthly_report_gen", morning_peak)
    print(f"Decision for 'monthly_report_gen' in morning: {report_task_decision_morning}")

    # Example with a custom solar prediction function
    def my_custom_solar_func(dt: datetime.datetime) -> float:
        # Very simple custom function: always 95% between 10 AM and 4 PM, else 20%
        if 10 <= dt.hour <= 16:
            return 95.0
        return 20.0

    custom_scheduler = SolarScheduler(solar_prediction_func=my_custom_solar_func)
    print("\n--- Testing with Custom Solar Prediction Function ---")
    custom_scheduler.decide_task_scheduling("high_priority_compute", now.replace(hour=11))
    custom_scheduler.decide_task_scheduling("low_priority_backup", now.replace(hour=20))
