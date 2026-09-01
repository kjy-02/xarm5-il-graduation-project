from glob import glob
from setuptools import find_packages, setup


package_name = "xarm_il"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Graduation Project Team",
    maintainer_email="student@example.com",
    description="xArm5 keyboard imitation-learning pipeline",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "keyboard_teleop = xarm_il.keyboard_teleop_node:main",
            "safety_mux = xarm_il.safety_mux_node:main",
            "episode_recorder = xarm_il.episode_recorder_node:main",
            "preflight_check = xarm_il.preflight_check:main",
            "policy_node = xarm_il.policy_node:main",
            "train_policy = xarm_il.train_policy:main",
            "validate_dataset = xarm_il.validate_dataset:main",
            "visualize_episode = xarm_il.visualize_episode:main",
            "smoke_test_pipeline = xarm_il.smoke_test_pipeline:main",
        ],
    },
)
