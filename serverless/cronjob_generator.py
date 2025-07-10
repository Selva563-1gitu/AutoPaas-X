import yaml
import os
from typing import Dict, Any, List

class CronJobGenerator:
    """
    Generates Kubernetes CronJob YAML manifests for serverless warm-up tasks
    or other scheduled operations.

    This class reads a base CronJob template and injects dynamic values
    like name, schedule, image, commands, and resource limits.
    """

    def __init__(self, template_path: str):
        """
        Initializes the CronJobGenerator.

        Args:
            template_path (str): The path to the base Kubernetes CronJob YAML template file.
        """
        self.template_path = template_path
        self.base_template = self._load_template()
        print(f"CronJobGenerator initialized with template: {template_path}")

    def _load_template(self) -> Dict[str, Any]:
        """
        Loads the base CronJob YAML template from the specified path.
        If the file does not exist, it creates a basic dummy template.

        Returns:
            Dict[str, Any]: The loaded (or created) YAML template as a dictionary.
        """
        if not os.path.exists(self.template_path):
            print(f"Warning: CronJob template not found at {self.template_path}. Creating a dummy template.")
            os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
            dummy_template = """apiVersion: batch/v1
kind: CronJob
metadata:
  name: placeholder-cronjob
  labels:
    app.kubernetes.io/name: autopaa-x-placeholder
    app.kubernetes.io/component: cronjob
spec:
  schedule: "0 0 * * *" # Default to midnight daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: default-container
            image: busybox
            command: ["/bin/sh", "-c", "echo 'Default cronjob running'"]
            resources:
              limits:
                cpu: "100m"
                memory: "128Mi"
          restartPolicy: OnFailure
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 1
  failedJobsHistoryLimit: 1
"""
            with open(self.template_path, 'w') as f:
                f.write(dummy_template)
            print(f"Dummy template saved to {self.template_path}")
            return yaml.safe_load(dummy_template)
        else:
            with open(self.template_path, 'r') as f:
                return yaml.safe_load(f)

    def generate_cronjob(self,
                         name: str,
                         schedule: str,
                         image: str,
                         command: List[str],
                         env_vars: Dict[str, str] = None,
                         resource_limits: Dict[str, str] = None,
                         labels: Dict[str, str] = None,
                         annotations: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Generates a Kubernetes CronJob manifest based on the loaded template
        and provided dynamic parameters.

        Args:
            name (str): The name of the CronJob.
            schedule (str): The cron schedule string (e.g., "0 7 * * *").
            image (str): The container image to use (e.g., "curlimages/curl:latest").
            command (List[str]): The command to execute in the container (e.g., ["/bin/sh", "-c", "curl http://my-app/health"]).
            env_vars (Dict[str, str], optional): Environment variables for the container. Defaults to None.
            resource_limits (Dict[str, str], optional): Resource limits (e.g., {"cpu": "100m", "memory": "128Mi"}). Defaults to None.
            labels (Dict[str, str], optional): Additional labels for the CronJob metadata. Defaults to None.
            annotations (Dict[str, str], optional): Annotations for the CronJob metadata. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary representing the Kubernetes CronJob manifest.
        """
        if not self.base_template:
            print("Error: Base CronJob template not loaded. Cannot generate manifest.")
            return {}

        # Deep copy to avoid modifying the base template directly
        manifest = self.base_template.copy()

        # Update metadata
        manifest['metadata']['name'] = name
        if labels:
            manifest['metadata']['labels'].update(labels)
        if annotations:
            manifest['metadata']['annotations'] = annotations # Overwrite or add

        # Update spec
        manifest['spec']['schedule'] = schedule
        job_template_spec = manifest['spec']['jobTemplate']['spec']['template']['spec']
        container = job_template_spec['containers'][0] # Assuming single container for simplicity

        container['name'] = f"{name}-container" # Ensure unique container name
        container['image'] = image
        container['command'] = command

        if env_vars:
            container['env'] = [{"name": k, "value": v} for k, v in env_vars.items()]
        else:
            # Ensure 'env' key is removed if no env_vars are provided
            if 'env' in container:
                del container['env']

        if resource_limits:
            container['resources'] = {
                "limits": resource_limits,
                "requests": resource_limits # Often requests are same as limits for simplicity
            }
        else:
            # Ensure 'resources' key is removed if no limits are provided
            if 'resources' in container:
                del container['resources']

        return manifest

    def save_cronjob_to_file(self, manifest: Dict[str, Any], output_dir="generated_k8s_manifests"):
        """
        Saves the generated CronJob manifest to a YAML file.

        Args:
            manifest (Dict[str, Any]): The CronJob manifest dictionary.
            output_dir (str): The directory to save the YAML file.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{manifest['metadata']['name']}.yaml"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        return file_path

# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Ensure manifests and generated_k8s_manifests directories exist
    os.makedirs("../manifests", exist_ok=True)
    os.makedirs("../generated_k8s_manifests", exist_ok=True)

    # Initialize generator, it will create a dummy template if not found
    generator = CronJobGenerator(template_path="../manifests/cronjob.yaml")

    print("\n--- Generating a Warm-up CronJob ---")
    warmup_cronjob = generator.generate_cronjob(
        name="my-app-warmup",
        schedule="*/15 * * * *", # Every 15 minutes
        image="curlimages/curl:latest",
        command=["/bin/sh", "-c", "curl -s -o /dev/null -w '%{http_code}' http://my-app-service.default.svc.cluster.local/health"],
        resource_limits={"cpu": "50m", "memory": "64Mi"},
        labels={"app": "my-app", "type": "warmup"},
        annotations={"description": "Periodic warm-up for my-app"}
    )
    print(yaml.dump(warmup_cronjob, default_flow_style=False, sort_keys=False))
    generator.save_cronjob_to_file(warmup_cronjob, "../generated_k8s_manifests")

    print("\n--- Generating a Daily Backup CronJob ---")
    backup_cronjob = generator.generate_cronjob(
        name="db-backup",
        schedule="0 2 * * *", # Every day at 2 AM
        image="myrepo/db-backup-tool:latest",
        command=["/bin/sh", "-c", "python /app/backup.py --db mydb --target s3://my-bucket"],
        env_vars={"DB_USER": "admin", "DB_PASS_SECRET": "my-secret"},
        resource_limits={"cpu": "200m", "memory": "256Mi"},
        labels={"app": "database", "job-type": "backup"}
    )
    print(yaml.dump(backup_cronjob, default_flow_style=False, sort_keys=False))
    generator.save_cronjob_to_file(backup_cronjob, "../generated_k8s_manifests")

    # Clean up generated_k8s_manifests directory for subsequent runs
    import shutil
    if os.path.exists("../generated_k8s_manifests"):
        print("\nCleaning up '../generated_k8s_manifests' directory...")
        shutil.rmtree("../generated_k8s_manifests")
