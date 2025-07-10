import yaml
import os

class DeploymentGenerator:
    def __init__(self, output_dir: str = "../generated_k8s_manifests"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_deployment(self, name: str, image: str, cpu_cores: float, ram_gb: float, port: int = 80,
                            replicas: int = 1, labels=None, annotations=None, command=None):
        if labels is None:
            labels = {"app": name}
        if annotations is None:
            annotations = {}

        cpu_millicores = int(cpu_cores * 1000)
        ram_mebibytes = int(ram_gb * 1024)

        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "labels": labels,
                "annotations": annotations
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": labels
                },
                "template": {
                    "metadata": {
                        "labels": labels,
                        "annotations": annotations
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": name,
                                "image": image,
                                "ports": [{"containerPort": port}],
                                "resources": {
                                    "limits": {
                                        "cpu": f"{cpu_millicores}m",
                                        "memory": f"{ram_mebibytes}Mi"
                                    }
                                },
                                "command": command
                            }
                        ]
                    }
                }
            }
        }

        return deployment

    def save_deployment_to_file(self, deployment_dict, filename_prefix: str):
        filename = f"{filename_prefix}_deployment.yaml"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            yaml.dump(deployment_dict, f)
        return filepath
