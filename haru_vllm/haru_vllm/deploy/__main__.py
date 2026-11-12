"""Print the path of the bundled deploy config: ``python -m haru_vllm.deploy``."""

from importlib import resources

print(resources.files("haru_vllm") / "deploy" / "haru.yaml")
